import numpy as np
import streamlit as st
from scipy.stats import norm
import matplotlib.pyplot as plt

# 复用上面核心函数
def calc_f2(ref, test):
    if len(ref) != len(test):
        raise ValueError("参比与受试时间点数量必须一致")
    n = len(ref)
    sum_sq = np.sum((ref - test) ** 2)
    f2 = 50 * np.log10(100 / np.sqrt(1 + sum_sq / n))
    return f2

def bca_bootstrap(data_ref, data_test, n_boot=20000, alpha=0.05, seed=42):
    np.random.seed(seed)
    n_ref = data_ref.shape[0]
    n_test = data_test.shape[0]
    mean_ref = np.mean(data_ref, axis=0)
    mean_test = np.mean(data_test, axis=0)
    f2_ori = calc_f2(mean_ref, mean_test)
    boot_f2 = []
    for _ in range(n_boot):
        idx_ref = np.random.randint(0, n_ref, size=n_ref)
        idx_test = np.random.randint(0, n_test, size=n_test)
        boot_ref = data_ref[idx_ref, :]
        boot_test = data_test[idx_test, :]
        b_mean_r = np.mean(boot_ref, axis=0)
        b_mean_t = np.mean(boot_test, axis=0)
        boot_f2.append(calc_f2(b_mean_r, b_mean_t))
    boot_f2 = np.array(boot_f2)
    z0 = norm.ppf(np.sum(boot_f2 < f2_ori) / n_boot)
    jack_f2 = []
    for i in range(n_ref):
        jk_ref = np.delete(data_ref, i, axis=0)
        jk_r_mean = np.mean(jk_ref, axis=0)
        jack_f2.append(calc_f2(jk_r_mean, mean_test))
    jack_f2 = np.array(jack_f2)
    jack_mean = np.mean(jack_f2)
    num = np.sum((jack_mean - jack_f2) ** 3)
    den = 6 * (np.sum((jack_mean - jack_f2) ** 2)) ** 1.5
    a = num / den if den != 0 else 0
    z_low = norm.ppf(alpha / 2)
    z_high = norm.ppf(1 - alpha / 2)
    q_low = norm.cdf(z0 + (z0 + z_low) / (1 - a * (z0 + z_low)))
    q_high = norm.cdf(z0 + (z0 + z_high) / (1 - a * (z0 + z_high)))
    ci_l = np.quantile(boot_f2, q_low)
    ci_u = np.quantile(boot_f2, q_high)
    return f2_ori, ci_l, ci_u, boot_f2

# 网页页面
st.set_page_config(page_title="BootstrapF2在线计算工具", layout="wide")
st.title("BootstrapF2在线分析工具")


col1, col2 = st.columns(2)
with col1:
    st.subheader("参比制剂溶曲数据")
    ref_text = st.text_area("每行1个样品，逗号分隔各时间点", value="12,35,62,83,97\n14,33,65,81,96\n11,37,60,85,98\n13,34,63,82,95\n12,36,61,84,97\n14,32,64,80,96")
with col2:
    st.subheader("受试制剂溶曲数据")
    test_text = st.text_area("每行1个样品，逗号分隔各时间点", value="13,34,61,82,96\n12,36,63,84,98\n14,33,60,81,95\n11,35,62,83,97\n13,32,64,80,96\n12,37,65,85,99")

n_boot = st.slider("Bootstrap抽样次数", min_value=5000, max_value=30000, value=20000, step=5000)
alpha = 0.05

# 解析输入数据
def parse_data(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    mat = []
    for line in lines:
        row = [float(v) for v in line.split(",")]
        mat.append(row)
    return np.array(mat)

if st.button("Bootstrap计算"):
    ref_mat = parse_data(ref_text)
    test_mat = parse_data(test_text)
    with st.spinner("正在计算（采样次数越大耗时越久）..."):
        f2_ori, ci_low, ci_high, boot_dist = bca_bootstrap(ref_mat, test_mat, n_boot, alpha)
    # 结果输出
    st.subheader("计算结果")
    st.write(f"Normal f2：{f2_ori:.2f}")
    st.write(f"95% CI：[{ci_low:.2f}, {ci_high:.2f}]")
    if ci_low >= 50:
        st.success("判定：置信下限≥50，溶出曲线相似")
    else:
        st.error("判定：置信下限＜50，溶出曲线不相似")

    # 绘制Bootstrap分布直方图
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(boot_dist, bins=60, alpha=0.7, color="#4488dd")
    ax.axvline(f2_ori, color="red", lw=2, label=f"Normal f2={f2_ori:.2f}")
    ax.axvline(ci_low, color="orange", ls="--", lw=1.5, label=f"95%CI lower limit={ci_low:.2f}")
    ax.axvline(ci_high, color="orange", ls="--", lw=1.5, label=f"95%CI upper limit={ci_high:.2f}")
    ax.axvline(50, color="green", ls=":", lw=2, label="Cutoff value 50")
    ax.set_xlabel("f2 value")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
