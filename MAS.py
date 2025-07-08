import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def logistic_univariate_and_multivariate(df, target, selected_columns):
    df = df.copy()
    significant_vars = []

    print("\n🔍 Hồi quy Logistic đơn biến (theo các biến chọn):")

    for col in selected_columns:
        try:
            if df[col].dtype == 'object' or df[col].nunique() < 5:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                for dummy_col in dummies.columns:
                    X = dummies[[dummy_col]].astype(float).dropna()
                    y = df[target].loc[X.index]
                    X = sm.add_constant(X)
                    model = sm.Logit(y, X).fit(disp=0)
                    coef = model.params[dummy_col]
                    p_value = model.pvalues[dummy_col]
                    print(f"{dummy_col:<35} | Coef: {coef:+.4f} | P-value: {p_value:.4f}")
                    if p_value < 0.05:
                        df[dummy_col] = dummies[dummy_col]
                        significant_vars.append(dummy_col)
            else:
                X = df[[col]].astype(float).dropna()
                y = df[target].loc[X.index]
                X = sm.add_constant(X)
                model = sm.Logit(y, X).fit(disp=0)
                coef = model.params[col]
                p_value = model.pvalues[col]
                print(f"{col:<35} | Coef: {coef:+.4f} | P-value: {p_value:.4f}")
                if p_value < 0.05:
                    significant_vars.append(col)
        except Exception as e:
            print(f"[!] Lỗi với biến {col}: {e}")

    print("\n✅ Biến có ý nghĩa thống kê (P < 0.05):", significant_vars)

    if significant_vars:
        print("\n📊 Kết quả hồi quy Logistic đa biến:")
        try:
            X_multi = df[significant_vars].dropna().astype(float)
            X_multi = sm.add_constant(X_multi)
            y = df[target].loc[X_multi.index]
            multi_model = sm.Logit(y, X_multi).fit()
            print(multi_model.summary())

            # 🧮 In công thức hồi quy
            coefs = multi_model.params
            equation = "logit(P(Survived)) = "
            equation += f"{coefs['const']:+.4f}"
            for var, coef in coefs.items():
                if var != 'const':
                    equation += f" + {coef:+.4f} * {var}"
            print("\n🧮 Công thức dự đoán:")
            print(equation)

            # 🔮 Dự đoán xác suất cho mẫu đầu tiên
            example = X_multi.iloc[0:1]
            prob = multi_model.predict(example)
            print(f"\n🔮 Xác suất dự đoán cho mẫu đầu tiên: {prob[0]:.4f}")

            # 🎯 Đánh giá độ chính xác mô hình với scikit-learn
            X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
            log_reg = LogisticRegression(max_iter=1000)
            log_reg.fit(X_train, y_train)
            accuracy = log_reg.score(X_test, y_test)
            print(f"🎯 Độ chính xác mô hình trên tập kiểm tra: {accuracy:.4f}")

        except Exception as e:
            print(f"[!] Lỗi hồi quy đa biến: {e}")

# ---------- CHẠY CHƯƠNG TRÌNH ----------
if __name__ == "__main__":
    df = pd.read_excel("MAS.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    target_col = "survived"

    selected_cols = [
        "age", "gender", "country", "family_history",
        "smoking_status", "bmi", "cholesterol_level",
        "hypertension", "asthma", "cirrhosis", "other_cancer"
    ]

    # Ép kiểu đúng với các cột binary
    for binary_col in ["hypertension", "asthma", "cirrhosis", "other_cancer"]:
        if binary_col in df.columns:
            df[binary_col] = pd.to_numeric(df[binary_col], errors='coerce')

    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    logistic_univariate_and_multivariate(df, target_col, selected_cols)
# 📈 Tính tỷ lệ sống sót
if target_col in df.columns:
    total = df[target_col].dropna().shape[0]
    survived_count = df[target_col].dropna().sum()
    survival_rate = survived_count / total * 100
    print(f"\n📈 Tỷ lệ sống sót khi mắc bệnh: {survival_rate:.2f}% ({survived_count}/{total})")
