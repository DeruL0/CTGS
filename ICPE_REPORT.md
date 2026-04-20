
# 1. CT値

重複するところでCT値について、基本的には符号付き距離 d(x) を計算しているので、それを使って空間を分割
$$
w_s(x) = \sigma\!\left(\frac{-d(x)}{\sigma_{\text{band}}}\right), \qquad w_b(x) = 1 - w_s(x)
$$

$$
\mathrm{CT}(x) = w_s(x)\cdot\rho_s(x) + w_b(x)\cdot\rho_b(x)
$$

$d(x)$ は点 $x$ から材料境界までの符号付き距離（内部を正、外部を負）。境界に近いほど $w_s \to 1$ で surface が支配的になり、内部では $w_b \to 1$ で bulk が支配的になる。


---

## 2. 「2DGS」と「3DGS」

実際はすべて3DGSであり、共通の密度式を用いる、だからslice gaussianとbulk gaussianという名前をつかいました

$$
G_i(x) = \exp\!\left(-\tfrac{1}{2}\left\|R_i^\top(x-\mu_i)/s_i\right\|^2\right)
$$

二種類の primitive の違いは、スケールの初期化と拘束にある。

| 種別 | スケール初期化 | 幾何形状 |
|------|--------------|---------|
| Surface Gaussian | $s_i = (s_{\text{tang}},\; s_{\text{tang}},\; s_{\text{thick}})$、$s_{\text{thick}} \ll s_{\text{tang}}$ | 薄い円盤状 |
| Bulk Gaussian | $s_j = (r,\; r,\; r)$ | 等方的な球状 |

surface gaussian は法線方向のスケールを強く拘束した三次元 Gaussian として実装されている。

---

## 3. 訓練 loss と constraints

現在の実装では、毎 iteration で完全な三次元 volume 全体を dense に比較するのではなく、確率的にサンプリングした CT slice patch と、三次元空間上の sparse field samples を組み合わせて最適化する。

Gaussian 集合から得られる密度場を

$$
\rho_\theta(x)=\sum_i \alpha_i G_i(x)
$$

とおく。ここで $\alpha_i$ は opacity である。密度は occupancy-like な値

$$
o_\theta(x)=1-\exp(-\rho_\theta(x))
$$

に変換し、三次元 field supervision では

$$
\hat f_\theta(x)=2o_\theta(x)-1
$$

を用いる。

全体の training objective は次のように書ける。

$$
\mathcal{L}
=\lambda_{\text{slice}}\mathcal{L}_{\text{slice}}
+\lambda_{\text{field}}\mathcal{L}_{\text{field}}
+\lambda_{\text{ridge}}\mathcal{L}_{\text{ridge}}
+\lambda_{\text{thick}}\mathcal{L}_{\text{thick}}
+\lambda_{\text{tang}}\mathcal{L}_{\text{tang}}
+\lambda_{\text{op}}\mathcal{L}_{\text{op}}
+\lambda_{\text{bulk}}\mathcal{L}_{\text{bulk}}
+\lambda_{\text{overlap}}\mathcal{L}_{\text{overlap}}
+\lambda_{\text{mat}}\mathcal{L}_{\text{mat}} .
$$

### 3.1 Slice patch reconstruction loss

軸 $a\in\{x,y,z\}$、slice index $k$、patch 領域 $P$ をサンプリングし、各 pixel を三次元点 $x_{a,k,u,v}$ に変換する。rendered patch は

$$
\hat I_P(u,v)=\rho_\theta(x_{a,k,u,v})
$$

として計算される。loss は CT volume から取得した ground-truth patch $I_P$ との L1 と SSIM の組み合わせである。

$$
\mathcal{L}_{\text{slice}}
=0.8\|\hat I_P-I_P\|_1
+0.2\left(1-\mathrm{SSIM}(\hat I_P,I_P)\right)
$$

注意点として、この項は full slice ではなく slice patch に対する stochastic loss である。ただし patch を render するときは、patch と交差する support を持つ Gaussian も計算に入る。

### 3.2 Sparse volumetric field loss

material support と air / void 領域から三次元点集合 $X$ をサンプリングする。CT volume から得た smoothed intensity field を threshold で正規化した target field を $f_{\text{CT}}(x)$ とすると、

$$
\mathcal{L}_{\text{field}}
=\frac{1}{|X|}\sum_{x\in X}
\mathrm{SmoothL1}\left(\hat f_\theta(x), f_{\text{CT}}(x)\right)
$$

となる。この項は dense volume loss ではなく、support / air samples に基づく sparse volumetric supervision である。

### 3.3 Boundary ridge alignment

surface Gaussian の中心が CT の境界 ridge に乗り、法線方向も CT gradient field と一致するように拘束する。surface Gaussian の法線を $n_i$、CT gradient から得る境界強度を $b(\mu_i)$、境界法線を $n_{\text{CT}}(\mu_i)$、gradient magnitude の法線方向微分を $D_n(\mu_i)$ とすると、

$$
\mathcal{L}_{\text{ridge}}
=\frac{1}{N_s}\sum_{i\in S}
\left[
\mathrm{SmoothL1}(D_n(\mu_i),0)
+\max(0,\tau_b-b(\mu_i))
+1-\left|n_i^\top n_{\text{CT}}(\mu_i)\right|
\right]
$$

である。これは surface center を境界付近に置き、surface normal を実際の CT 境界法線に合わせるための幾何拘束である。

### 3.4 Surface constraints

surface Gaussian は薄い境界 primitive として振る舞う必要がある。そのため、法線方向厚み、接線方向スケール、opacity を正則化する。

法線方向 thickness は

$$
t_i=
\sqrt{
\sum_{m=1}^{3}
\left((R_i^\top n_i)_m s_{i,m}\right)^2
}
$$

で定義し、

$$
\mathcal{L}_{\text{thick}}
=\frac{1}{N_s}\sum_{i\in S}\max(0,t_i-t_{\max})
$$

を用いる。

接線方向 RMS scale は

$$
r^{\text{tang}}_i=
\sqrt{
\frac{\|s_i\|^2-t_i^2}{2}
}
$$

として、

$$
\mathcal{L}_{\text{tang}}
=\frac{1}{N_s}\sum_{i\in S}\max(0,r^{\text{tang}}_i-r_{\max})
$$

で過度な広がりを抑える。

opacity は saturation を避けるため、

$$
\mathcal{L}_{\text{op}}
=\frac{1}{N}\sum_i
\max(0,\alpha_i-\alpha_{\text{target}})^2
$$

で制約する。

### 3.5 Bulk constraints

bulk Gaussian は内部を埋めるための primitive であるが、過度に大きくなると cavity や thin structure を潰す可能性がある。そのため、material 内部の distance transform から bulk scale の上限を作る。

$$
s^{\max}_i
=\mathrm{clamp}
\left(
\eta\,\mathrm{EDT}(\mu_i),
s_{\min},
s_{\text{bulk,max}}
\right)
$$

$$
\mathcal{L}_{\text{bulk}}
=\frac{1}{N_b}\sum_{i\in B}
\max(0,\max_m s_{i,m}-s^{\max}_i)^2
$$

また、隣接 bulk Gaussian が過度に重なることを避けるため、

$$
\mathcal{L}_{\text{overlap}}
=\frac{1}{|\mathcal{N}|}
\sum_{(i,j)\in\mathcal{N}}
\max(0,r_i+r_j-\|\mu_i-\mu_j\|)
$$

を用いる。実装ではさらに optimizer step 後に bulk scale を hard projection で上限以下に clip する。またデフォルトでは bulk Gaussian の中心位置は固定し、scale と opacity を中心に調整する。

### 3.6 Material boundary constraint

複数 material label がある場合、異なる material に属する近傍 surface Gaussian pair に対して、境界付近の opacity が低くなりすぎないようにする。

$$
\mathcal{L}_{\text{mat}}
=
\mathbb{E}_{(i,j):m_i\neq m_j}
\left[
\max(0,\alpha_{\text{mat}}-\min(\alpha_i,\alpha_j))
\exp\left(-\frac{\|\mu_i-\mu_j\|}{\bar d}\right)
\right]
$$

これにより、material 間境界の surface representation が弱くなることを防ぐ。

---

## 4. アブストラクト

### CTGS: Hybrid Gaussian Splatting for Compact Industrial CT Representation

Computed tomography (CT) is widely used for inspecting complex manufactured parts in industrial settings, but existing CT representations often struggle to balance fast visualization, compact storage, and geometry-aware analysis. A practical representation should therefore support efficient rendering while preserving sufficient geometric fidelity for downstream tasks.

We propose CTGS, a hybrid Gaussian representation tailored for industrial CT volumes. CTGS employs two complementary primitives: surface Gaussians, which are thin and anisotropic, aligning with material surfaces, and bulk Gaussians, which are isotropic, filling material interiors. Given a reconstructed CT volume as input, CTGS first separates material regions from void, extracts material surfaces, and initializes both primitives accordingly. We then train the model with a slice-based reconstruction loss, material-aware occupancy supervision, and geometric regularization terms that preserve surface geometry and internal structures.

CTGS achieves significant data reduction relative to the original CT volume for lightweight display, while surface Gaussians retain material surface accuracy for downstream analysis.
