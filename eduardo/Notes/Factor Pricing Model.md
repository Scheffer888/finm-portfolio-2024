
# Factor Pricing Models

## 1. Introduction

Linear Pricing Model is a model that try to explain the *long-run expected excess return (expected return above the risk-free rate)* of an asset as a linear function of its exposure (beta) to certain risk factors, each having an expected excess returns associated with it.

A *Factor Pricing Model*:
- Explains how expected excess returns vary accross assets (not how they vary across time).
- Is all about **expected** excess returns (not realized returns).
- Models the long-term expected returns of an asset.

The **"risk premium"** is the expected excess return of a factor due to the risk it represents.
- *The risk premium of an asset is proportional to the risk premium of the factors it is exposed to.*

*It Very different from Linear Factor Decomposition*, which is about realized returns and how they vary across time.

The intuition behind factor investing is that a standard diversification strategy emphasizing asset classes may lack optimal exposure to factors known to yield high average returns over time.

Moreover, since diversification involves managing the correlation structure across investments, factor investing can enhance diversification by providing a richer set of cross-correlations, better managing the risk and return of a portfolio across market cycles.

---

### 1.1. Risk Premium (Rational vs. Behavioral)

- Rational Risk Premium: The expected excess return of a factor due to the risk it represents.
- Behavioral Risk Premium: The expected excess return of a factor due to the mispricing of the factor.

- There is still a lot of debate nowadays whether it's in compensation for risk or whether the risk premium might just be due to markets being irrational (ex. bubbles and trends).

- The risk premium is the expected excess return of a factor, that it, the rewardfor holding an specific factor risk.

---

### 1.2. Beta

The **beta** of an asset with respect to a factor is the sensitivity of the asset's expected excess return to the factor.

*We got those betas from the Linear Factor Decomposition (time-series regression) and we are using them to predict the expected excess return of the asset.*

- A negative market beta mean that the asset is expected to have a lower return when the market goes up (negative mean excess return). Ex.: put options.
- The betas are not typical time series regression betas (like in a Linear Factor Decomposition), because if they were, the equation E[r_i] = beta_i,m * E[r_m] would actually be E[r_t] = alpha + beta_m,t * E[r_m,t] + epsilon_t
- Whatever happens this month (ex. stock goes down while the market goes up), it doesn't matter. This Factor Pricing Models are a theory about expected long-term average excess returns.
- Different than Linear Factor Decomposition, where the the left and right hand side are about realized returns. Here we are talking about expected excess returns.
    - In a Linear Factor Decomposition, the beats means that, on average, when the market goes up by 1%, the asset goes up by beta_i,m %.

- **Market Beta:** The sensitivity of a stock's return to the market return.
- **Size Beta:** The sensitivity of a stock's return to the size factor.
- **Value Beta:** The sensitivity of a stock's return to the value factor.

We get each of those betas by running a regression of the stock's return on the factors' return.

- The factors don't have to have a trend over time if the market is trending over time, because they are hedged long-short portfolios: they could go up a lot or down a lot.

What matters is the security's **beta** matters, not its measure of the **characteristic**.

- *The premium comes if the stock "acts like" the factor, not if the stock "has" the characteristic.*
- So what does the FF model expect of a stock with high B/M yet low correlation to other high B/M stocks?
- **Beta** earns premium—not the stock’s characteristic.
- This is one difference between FF "value" investing and Buffett-Graham "value" investing.

---

### 1.3. Smart Beta ETFs

**Strategy:**
- A smart beta ETF is a type of exchange-traded fund that uses alternative index construction rules to traditional market capitalization-based indices.
- The strategy is to use factor models to systematically select, weight, and rebalance stocks in the ETF to add value by tilting its exposure to specific risk premia.
- Smart beta portfolios are constructed such that the emphasis is weighting stocks in these portfolios not on the traditional measure of market capitalization, but by incorporating into their weighting scheme some aspect of a security’s fundamental value, such as a stock’s B/M ratio, profitability, or a characteristic of the security’s performance, such as a stock’s momentum.


Smart beta ETFs are considered a combination of passive and active investing:
- They are **passive** because they passively mimic what are termed **factor indexes** and hence do not require any input from a portfolio manager.
- They are **active** because their weights deviate from standard market capitalization weights.

**Advantages:**
- Lower fees than active management and hedge funds.
- Transparent and rules-based.
- Diversified and liquid.

**Disadvantages:**
- Do not capture the pure factors because ETFs are not allowed to short stocks, so the long-only portfolios end up having high correlation with the market (not "near-zero" market beta).

---
---

## 2. CAPM

The Capital Asset Pricing Model (CAPM) is a linear factor model that relates the expected excess return of an asset to its beta with the market portfolio and the market risk premium.

The CAPM identifies the **market portfolio** as the tangency portfolio.

- The market portfolio is the value-weighted portfolio of all available assets.
- It should include every type of asset, including non-traded assets.
- In practice, a broad equity index is typically used.
- *In practice, we use a broad equity index, like SPY or Russell 1000.*

The CAPM is about **expected** returns:

- The expected return of any asset is given as a function of two market statistics: the risk-free rate and the market risk premium.
- The coefficient is determined by a regression. 
- The theory does not say anything about how the risk-free rate or market risk premium are given: it is a **relative pricing formula**.

*Market beta is the only risk associated to higher average returns:*
- No other characteristics of the asset command a higher expected excess return returns from investors
- **Beyond how it affects market beta, CAPM says volatility, skewness, and other covariances do not matter** for determining risk premia.
- Idiosyncratic risks have such a negligible effect on the portfolio that in the limit it becomes meaningless  (what matters is its covariance with the rest of the portfolio).
- The only thing that matters is the beta.


---

### 2.1. Derivation of the CAPM


First method: If returns have a joint normal distribution...

1. The mean and variance of returns are sufficient statistics for the return distribution.
2. Thus, every investor holds a portfolio on the MV frontier.
3. Everyone holds a combination of the tangency portfolio and the risk-free rate.
4. Then aggregating across investors, the market portfolio of all investments is equal to the tangency portfolio.

However:
1. Returns are not jointly normal nor iid.
2. Investors ae not each holding a portfolio on the MV frontier.
3. Then. in reality, the market portfolio is not the tangency portfolio.

Second Method: Don’t assume the returns are jointly normal.

- This is another way of assuming all investors choose MV portfolios (only care about mean and variance of return).
- But now it is not because mean and variance are sufficient statistics of the return distribution, but rather that they are sufficient statistics of investor objectives.
- So one derivation of the CAPM is about return distribution, while the other is about investor behavior.

However:
- Investors do not all hold MV portfolios, some could be holding inefficient portfolios based on their individual preferences or what makes them feel good.


---


### 2.2. Return Variance Decomposition

The CAPM implies a clear relation between volatility of returns and risk premia.

- Consider the linear factor decomposition:
$$\tilde{r}_t^i = \beta^{i,m} \tilde{r}_t^m + \epsilon_t$$

- Take the variance of both sides of the equation to get:
$$\sigma_i^2 = (\beta^{i,m})^2 (\sigma^m)^2 + \sigma_\epsilon^2$$
$$\sigma_i^2 = sistemaic + idiosyncratic$$

So CAPM implies...

- The variance of an asset’s return is made up of a systematic (or market) portion and an idiosyncratic portion --> *Only the former risk is priced.*


---


### 2.3. Proportional Risk Premium

The CAPM implies that the risk premium of an asset is proportional to the risk premium of the market.

$$\mathbb{E} \left[ \tilde{r}^i \right] = \beta^{i,m} \, \mathbb{E} \left[ \tilde{r}^m \right]$$

- Using the definition of \( \beta^{i,m} \):

$$\frac{\mathbb{E} \left[ \tilde{r}^i \right]}{\sigma^i} = \left( \rho^{i,m} \right) \frac{\mathbb{E} \left[ \tilde{r}^m \right]}{\sigma^m}$$


**The CAPM and Sharpe Ratio:**

Using the definition of the Sharpe ratio in (3), we have

$$\text{SR}^i = \left( \rho^{i,m} \right) \text{SR}^m$$

- The Sharpe ratio earned on an asset depends only on the correlation between the asset return and the market.

- A security with large idiosyncratic risk, $( \sigma_\epsilon^2 )$, will have lower $( \rho^{i,m} )$, which implies a lower Sharpe Ratio.

- *If there is a factor (or combination of factors) such that it has the highest sharpe ratio, then this factor (or this combination of factors) is the tangency portfolio.*

- The math shows that all assets have a sharpe ration smaller or equal to the market sharpe ratio (because correlation is between -1 and 1).

- Thus, risk premia are determined only by systematic risk.

*All securities must have a Sharpe ratio smaller than the market Sharpe ratio (which returns to the idea that market portfolio is the tangency portfolio).*


**The CAPM and Treynor Ratio:**

$$\text{Treynor Ratio} = \frac{\mathbb{E} \left[ \tilde{r}^i \right]}{\beta^{i,m}}$$

- If CAPM does not hold, then Treynor’s Measure is not capturing all priced risk.
- If the CAPM does hold, then: *Treynor Ratio should be equal to all securities = market portfolio risk premium.*
- If we calculated the tangency portfolio using MV and expected returns for each asset by its mean return, and use that portfolio as the market portfolio, then the Treynor Ratio would be the same for all assets.
    - But we cannot do that because we don't know the expected returns of the assets - and that's exactly what we are trying to find out.


---


### 2.4. CAPM as Practical Model

For many years, the CAPM was the primary model in finance.

- In many early tests, it performed quite well.
- Some statistical error could be attributed to difficulties in testing.
- For instance, the market return in the CAPM refers to the return on all assets—not just an equity index (*Roll critique*): you can't disproved the CAPM by using an equity index because you never used the true market portfolio.
- Further, working with short series of volatile returns leads to considerable statistical uncertainty.


---
---

## 3. Fama-French Three-Factor Model


The **Fama-French 3-factor model** is one of the most well-known multifactor models.

$$\mathbb{E} \left[ \tilde{r}^i \right] = \beta^{i,m} \mathbb{E} \left[ \tilde{r}^m \right] + \beta^{i,s} \mathbb{E} \left[ \tilde{r}^s \right] + \beta^{i,v} \mathbb{E} \left[ \tilde{r}^v \right]$$

- $\tilde{r}^m$ is the excess market return as in the CAPM.
- $\tilde{r}^s$ is a portfolio that goes long small stocks and shorts large stocks.
- $\tilde{r}^v$ is a portfolio that goes long value stocks and shorts growth stocks.


---


### 3.1. Value Factor:

Different investors can measure value in different ways.

For Fama and French, the **book-to-market (B/M) ratio** is the market value of equity divided by the book (balance sheet) value of equity.

- High B/M means strong (accounting) fundamentals per market-value-dollar.
- High B/M are **value stocks**.
- Low B/M are **growth stocks**.

*Low*: < 30% percentile  
*High*: > 70% percentile

For portfolio value factor, this is the most common measure.

**Other Value Measures:**
Many other measures of value are based on some cash-flow or accounting value per market price.

- **Earnings-price** is a popular metric beyond value portfolios. Like B/M, the E/P ratio is accounting value per market valuation.
- **EBITDA-price** is similar, but uses an accounting measure of profit that ignores taxes, financing, and depreciation.
- **Dividend-price** uses common dividends, but is less useful for individual firms as many have no dividends.

Many other measures exist, with competing claims to being a special/better measure of "value."


#### 3.1.1. Value vs. Growth Stocks:
The labels "growth" and "value" are widely used.

- Historically, value stocks have delivered higher average returns.
- So-called "value" investors try to take advantage of this by looking for stocks with low market price per fundamental or per cash-flow.
- Much research has been done to try to explain this difference of returns and whether it is reflective of risk.

**Growth Stocks:**
Stocks trading at a low price relative to a measure of fundamental value such as book equity.
- It doesn't mean they have actually grown a lot in the past. It means that it doesn't have much accounting value per stock.
- The name comes from the implication that investors must be expecting the company to grow a lot in the future to justify paying a high price for the stock relative to its book value (low book to market ratio).

**Value Stocks:**
- High book-to-market ratio companies are generally less profitable and are relatively distressed, so these firms are riskier and have a higher average return.


#### 3.1.2. Construction of Value Factor:
- They consider a relative size:
    - *Long top 30% of stocks highest book-to-market ratios*
    - *Short bottom 30% of stocks with lowest book-to-market ratios.*
- Reason to consider a relative size and long-short portfolio:
    - Hedge out 


 ---


### 3.2. Size Factor:


#### 3.2.1. Construction of Size Factor:
- Consider a relative size:
    - *Long top 30% of stocks with smallest market capitalization*
    - *Short bottom 30% of stocks with large market capitalization*
- Reason to consider a relative size and long-short portfolio:
    - Hedge out the market beta by constructing a long-short portfolio.
    - Then long the Small ones and short the Large ones dolar for dolar.
    - Market beta is not gonna be hedged to zero with the dolar for dolar long-short portfolio, but it will be very closse to zero.

*Small*: < 30% percentile  
*Big*: > 70% percentile

---
---

## 4. Other Popular Factors:

- **5-Factor Model**: Extends the 3-Factor Model by adding **Profitability (RMW)** and **Investment (CMA)** factors.
    - Some argue that the investment factor is redundant, if measured value factor differently.
    - Still, one should be careful to drop HML. Prior to doing that, it is necessary to check the cross-sectional test with and without the Value factor and calculate the weights of the tangency portfolio. If HML shows relevant results, it should not be dropped and, thus, is not redundant.

Sort portfolios of equities based on...

- **Price movement**: Momentum, mean reversion, etc.
- **Volatility**: Realized return volatility, market beta, etc.
- **Profitability***: Robust-minus-Weak.
- **Investment***: Conservative-minus-Aggressive: measures how much a firm reinvests cash back into the firm (e.g., retained earnings or dividends).
    - Long low reinvestment firms and short high reinvestment firms.
  
*As measured in financial statements.

### 4.1. Construction:

- Always consider long-short portfolio to hedge out market beta to reduce its correlatoin with the market factor and achieve a better statistical power

- Fama and French use a simple approach of sorting stocks into deciles based on the factor measures and then taking the difference in returns between the top 30% and bottom 30% (leaving out the middle 40%) to calculate each factor's return, but one could use more sophisticated methods.


---
--- 


## 5. Momentum Factor:

**Return Autoregressions: Momentum and Reversion**

$$ r_{t+1}^m = \alpha + \beta r_t^m + \epsilon_{t+1} $$

The autoregression does not find $\beta$ to be significant (statistically or economically).

- **Positive $\beta$**: momentum.
  - An above average return in $t+1$ tends to relate to an above average return in $t$.
- **Negative $\beta$**: mean reversion.

We can write this regression as

$$ (r_{t+1}^m - \mu) = \beta (r_t^m - \mu) + \epsilon_{t+1} $$

where $\mu$ is the mean of $r^m$, and $\alpha = (1 - \beta) \mu$.


**Autocorrelation in the overall Market Index:**

- With the overall market index, there is no clear evidence of momentum or mean-reversion.
- Momentum on S&P: near zero.

**Autocorrelation in Individual Stocks**

- At a monthly level, most equities would have no higher than $\beta = 0.05$.
- Thus, for a long time, the issue was ignored; too small to be economical—especially with trading costs!

---

### 5.1. Trading on Small Autocorrelation

Two keys to taking advantage of this small autocorrelation:

1. **Trade the extreme “winners” and “losers”** (Select extreme)
   - Small autocorrelation multiplied by large returns gives sizeable return in the following period.
   - By additionally shorting the biggest “losers,” we can magnify this further.

2. **Hold a portfolio of many “winners” and “losers.”** (Diversify)
   - By holding a portfolio of such stocks, diversifies the idiosyncratic risk.
   - Very small $R^2$ stat for any individual autoregression, but can play the odds (i.e., rely on the small $R^2$) across 1000 stocks all at the same time. (*go from 1% to 6%*)

---

**Illustration: Workings of Momentum**

- Assume each stock $i$ has returns which evolve over time as

  $$ \left( r_{t+1}^i - \underbrace{0.83\%}_{\text{mean}} \right) = \underbrace{0.05}_{\text{autocorr}} \left( r_t^i - \underbrace{0.83\%}_{\text{mean}} \right) + \epsilon_{t+1} $$

- Assume there is a continuum of stocks, and their cross-section of returns for any point in time, $t$, is distributed as

  $$ r_t^i \sim \mathcal{N} (0.83\%, 11.5\%) $$

---

**Illustration: Normality**

From the normal distribution assumption:
- *The top 10% of stocks in any given period are those with returns greater than $1.28 \sigma$.*
- Thus, the mean return of these “winners” is found by calculating the *conditional mean*:
- For a normal distribution, we have a closed form solution for this conditional expectation (mean of a truncated normal):

  $$ \mathbb{E} \left[ r \mid r > 1.28\sigma \right] = 1.755 \sigma = 21.01\% $$

  *[Same math as CVaR]*

---

**Illustration: Autocorrelation**

From the autocorrelation assumption:

- A portfolio of time $t$ winners, $r^u$, is expected to have a time $t+1$ mean return of

  $$ \mathbb{E}_t \left[ r_{t+1}^u \right] = 0.83\% + .05 (1.755\sigma - 0.83\%) = 1.84\% $$

- We assumed that the average return across stocks is 0.84%.
- Thus, the momentum of the winners yields an additional 1% per month.
- Long the winners + short the losers --> 2x excess return.

### 5.2. In Practice:

**Trading Costs:**
- Maybe if you have a stock that was part of the long portfolio in the previous month, and now barely makes out of it, then it might be better not to remove it from the long portfolio.
- Similarly, if a new stock barely makes it into the long portfolio, then it might be better not to add it to the long portfolio.

**High turnover:**
- To decrease turnover, take the biggest winners from the past 12 months.
- In the next month, the biggest winners from the past 12 months will still have a high turnover, but manageable.

**Tax considerations:**


Trade-off: we want concentration on the extremes (top low 1%), but then we want to diversify across many stocks (risk-return trade-off).


---  

### 5.3. Explanations for Momentum


**Risk-Based Explanations**

Is the momentum strategy associated with some risk?

- *Volatility?*
- *Correlation to market index, such as the S&P?*
- *Business-cycle correlation?*
- *Tail risk?*
- *Portfolio rebalancing risk?*


**Behavioral Explanations**

Can investor behavior explain momentum?

- *Under-reaction to news*
  - At time $t$, positive news about stock pushes price up 5%.
  - At time $t + 1$, investors fully absorb the news, and the stock goes up another 1% to rational equilibrium price.
  
- *Over-reaction to news*
  - At time $t$, positive news about stock pushes price up 5%—to rational equilibrium.
  - At time $t + 1$, investors are overly optimistic about the news and recent return. Stock goes up another 1%.

---
---


## 6. Economic Factors (CCAPM)

*Main objective:* it provides economic reasoning on where you should look for new factors


### 6.1. Non-Return Factors

What if we want to use a vector of factors, $\mathbf{z}$, which are not themselves assets?

- Examples: slope of the term structure of interest rates, liquidity measures, economic indicators (consumption, unemployment data), etc.
- The time-series tests of Linear Factor Models (LFM) relied on:
  $$ \lambda_z = \mathbb{E} \left[ \tilde{r}^z \right], \quad \alpha = 0 $$
- However, with untraded factors, $\mathbf{z}$, we do not have either implication.
- To test an LFM with untraded factors, we must perform a **cross-sectional test**.

**Other examples:**
- Investors care about the market going down, e.g. tail risk
- What correlations do investors not like (i.e., what are they really adverse to)?
  - Perhaps it's not as much about a slight market downturn, but a larger loss, like a 20% decline.
  - Or perhaps they are adverse to lifestyle changes rather than simple market movements.

### 6.2. The Consumption CAPM (CCAPM)

The Consumption CAPM (CCAPM) suggests that the **only systematic risk is consumption growth**.
- You don't want an investment positively related to consumption (investment goes down when your consumption goes down); you want the opposite.

$$ \mathbb{E} \left[ \tilde{r}^i \right] = \beta^{i,c} \lambda_c $$

where $c$ represents some measure of consumption growth.

**Challenges**:
  - Specifying an accurate measure for $c$.
- The CAPM can be seen as a special case where $c = \tilde{r}^m$.
- Typically, measures of $c$ are **non-traded factors**.
- We could build a **replicating portfolio** or test it directly in the cross-section.


### 6.3. Testing the CCAPM Across Assets

*We cannot run a time-series test because consumption is not an asset; we must run a **cross-sectional test to reveal $\lambda_c$**.

1. **Time-Series Regression**:
   - Run the regression for each test-security, $i$:
     $$ \tilde{r}_t^i = a^i + \beta^{i,c} c_t + \epsilon_t^i $$
   - Here, the intercept is denoted $a$ to emphasize it is not an estimate of model error, $\alpha$.
   - The time series $alpha$ in this regression is meaningless.

2. **Cross-Sectional Regression**:
   - Run a single cross-sectional regression to estimate the premium, $\lambda_c$, and the residual pricing errors, $\alpha^i$:
     $$ \mathbb{E} \left[ \tilde{r}^i \right] = \lambda_c \beta^{i,c} + \alpha^i $$
   - Theory implies that the cross-sectional regression should not have an intercept, though it is often included in practice.

### 6.4. Macro Factors

A number of industry models use **non-traded, macro factors**.

- *GDP growth*
- *Recession indicator*
- *Monetary policy indicators*
- *Market volatility*

The Economic theory says the factors should only work if:
  - it is correlated to things investors are risk averse about (if there is a rational pricing);
  - if it has nothing to do with risk aversion - if there is an irrational, behavioral bias;
  - or maybe it's about inefficiencies in the market.


*Note*: Consumption factors are widely studied in academia but less so in the industry.

  *Economic factors should be checked to see if they align with investor risk aversion.*

---
---


## 7. Factor Timing

**Size and Value Factors:**

- The returns and risks of size and value factors are highest in the early part of an economic expansion
- Outperform when rates are rising. 

**Momentum and Quality Factors:**
- Perform best at the start of an economic contraction
- Outperform in declining interest rate environments

These observations suggest that factor investment portfolios can manage return and risk by combining factors with different cyclicality, thereby mitigating the effects of changing business conditions.

Given the strong cyclicality in factor returns, investors may consider switching between factor portfolios in response to anticipated economic conditions to enhance returns. This practice is known as **style** or **factor timing**.


**Cliff Asness (AQR):** found "timing strategies to be quite weak historically.
- Rather than attempting to time factors, he recommends that investors "instead focus on identifying factors that they believe in over the very long haul, and aggressively diversify across them."

**Robert Arnott (Research Affiliates):**  "active timing of smart beta strategies and/or factor tilts can benefit investors.
-  We find that performance can easily be improved by emphasizing the factors or strategies that are trading cheap relative to their historical norms and by deemphasizing the more expensive factors or strategies."


---
---


## 8. Testing of Linear Pricing Models

**Outline**

- If you had a 5-factor model, then you could still combine those into a single factor and return to the same model (but would not be a "market factor," but a "combination factor").
- You might use a 5-factor model if those factors are clearly identifiable.
  - Then, you are saying that the market portfolio is not the tangency portfolio, but the combination of those factors is a tangency portfolio.

**Testing**

- We need to find the factor returns without using the expected returns of the assets, i.e., without calculating the tangency portfolio explicitly (circularity).
  - Find ways/assumptions to calculate the factor.

- The difficulty is *knowing* you have the correct model.
- Calculating the model, once you have it, is easy.
- We will need to test it (involving regression).

---

### 8.1. Time-Series Test: CAPM and Realized Returns

Here we focus on CAPM, but we can use with any linear pricing model.

The CAPM implies that expected returns for any security are

$$\mathbb{E} \left[ \tilde{r}^i \right] = \beta^{i,m} \, \mathbb{E} \left[ \tilde{r}^m \right]$$

This implies that realized returns can be written as

$$\tilde{r}_t^i = \beta^{i,m} \tilde{r}_t^m + \epsilon_t$$

where $\epsilon_t$ is *not* assumed to be normal, but: $\mathbb{E} \left[ e \right] = 0$


**Testing the CAPM on an Asset**

- Run a time-series regression of excess returns $i$ on the excess market return.
- Regression for asset $i$, across multiple data points $t$:

  $$\tilde{r}_t^i = \alpha^i + \beta^{i,m} \tilde{r}_t^m + \epsilon_t^i$$

  Estimate $\alpha$ and $\beta$.

**Alpha must be zero if CAPM holds:** $\alpha^i = 0$.
- Even if the true population $\alpha$ is zero, the sample $\alpha$ might not be zero.
- Check if $\alpha$ is "close enough" to zero:
    - P-test would check if the true population $\alpha$ should be zero.
- However testing on expectations is hard because they are not very precise.

*we know that $\alpha$ should be zero for all assets, so we perform a joint test on all $\alpha^i$:* chi-squared test.
- Interpretation of the **joint test**:
    - run the best (mean-variance) portfolio of alphas and epsilons and hedge out the market factor.
    - Then calculate the sharpe ratio. This sharpe ratio should be zero because we hedged out the market factor, so there is no premium.
    - By performing multiple t-tests simultaneously increases the likelyhood of finding at least one significant result purely by chances.
    - The H-test, on the other hand, combines all the t-tests into a single test accounting for the alphas joint distribution, reducing the probability of encountering a deviation from the CAPM model purely by chance.

- Zero Alpha implies that the risk factor completely captures everything (implying the risk factor *is* the tangency portfolio)
- Non-zero Apha implies that the risk factor is not 100% responsible for the expected excess return of the portfolios.
- In other words, we are assessing whether the risk factors replicate (or span) our tangency portfolio.
    
- If the model explains premium well, there should be no alpha.

**Pricing error ($\alpha$):** 

- The alphas are the errors in the model (the difference between the expected return, based on the betas and the market risk premium, and the realized return).

- They are the mean returns that the factors *cannot explain*

- We get one alpha per asset.

- To compare across assets, we can take the *mean absolute error (MAE)* of the alphas.
    - We expect the MAE to be zero if the model is correct.

*(Note: in a hedging regression, the error is the residual, not the alpha)*


**$R^2$ of the CAPM do not matter**

- I'm using the betas from the Linear Factor Decomposition to predict the expected returns of the asset.
- I'm using the alphas from the Linear Factor Decomposition to test the CAPM
- I am doing nothing with the $R^2$ of the regression.
- The $R^2$ of the regression is saying:
    - Is this a good hedge?
    - Is this a good replication/ decomposition?
    - It does not say anything about the quality of the Linear Pricing Model (or the expected long term  excess returns).
    - An $R^2$ only shows how good a factor decomposition is → used to find models statistically (when you don’t know where to look).
    - You might have a good factor model (good pricing prediction) and low $R^2$, and vice versa.
    - Even if the CAPM were exactly true, it would not imply anything about the $R^2$ of the above

- CAPM only cares about risk premia of the asset compared to the factor risk premia.
- CAPM does not say "stock A is down this month because the market is down" (that's a realized return).
- CAPM explains variation in $\mathbb{E} \left[ \tilde{r}^i \right]$ across assets—not variation in $\tilde{r}^i$ across time!

$$\tilde{r}_t^i = \alpha^i + \beta^{i,m} \tilde{r}_t^m + \epsilon_t$$


**In summary:**
we run a time series regression on the Linear Factor Decomposition.
    - Unlike heding  or replication, we don't care about the R-squared of the regression here.
    - We care about the $alphas$. If the $alphas$ are statistically different than zero, then we have a pricing error.
    - Statistical significance through chi-squared test of alphas.
    - The $alphas$ are the pricing errors, and they should be zero if the model is correct.

---


### 8.2. Cross-Section OLS Regression test: Industry Portfolios

A famous test for the CAPM is a collection of industry portfolios.
Consider a graph of the mean excess return of each industry portfolio against its market beta.

- Stocks are sorted into portfolios such as manufacturing, telecom, healthcare, etc.
- Variation in mean returns should be accompanied by variation in market beta.
    - Betas were obtained from a regression (linear factor decomposition).
- All portfolios should fit within the line if CAPM holds for $r^m$.

**Cross-Section Regression:**

Objective now is to find how close the portfolios fit the line passing through the origin and the market portfolio in a (market beta x historical excess return) graph --> **Security Market Line** (SML).

$$\mathbb{E} \left[ \tilde{r}^i \right] = \underbrace{\eta}_{\alpha} + \underbrace{\beta^{i,m}}_{x^i} \underbrace{\lambda_m}_{\beta^j} + \underbrace{v^i}_{\epsilon^i}$$

- Aach asset is a data point in the regression.
- The data on the left side is a list of mean returns on assets, $\mathbb{E} \left[ \tilde{r}^i \right]$.
- The data on the right side is a list of asset betas: $\beta^{i,m}$ for each asset $i$.
- The regression parameters are $\eta$ and $\lambda_m$.
- The regression errors are $v^i$.

**In summary:**
We run a regression of the cross section of stock returns on the factor betas
- Look at the R-squared of the regression. The higher the R-squared, the better the model.
- The error term here is the residual return, which is the return that is not explained by the factors.


---


### 8.3. Tangency Portfolio test:
- Calculate the tangency portfolio, which is the portfolio that maximizes the Sharpe ratio.
- If the tangency portfolio does not relevant weights on some of the factors, then it means that that factor is unimportant relative to the others and could be dropped.


---


**The Risk-Return Tradeoff**

To check that the *slope of the SML is the market risk premium*, note that the CAPM can be separated into two statements:

- Risk premia are proportional to market beta:

   $$\mathbb{E} \left[ \tilde{r}^i \right] = \beta^{i,m} \lambda_m$$

- The proportionality is equal to the market risk premium:

   $$\lambda_m = \mathbb{E} \left[ \tilde{r}^m \right]$$
   
The parameter $\lambda_m$ is the slope of the line.

- It represents the amount of risk premium an asset gets per unit of market beta.
- Thus, can divide risk premium into quantity of risk, $\beta^{i,m}$, multiplied by *price of risk*, $\lambda_m$.


---


**Cross-Section Test of the CAPM:**

$$\mathbb{E} \left[ \tilde{r}^i \right] = \eta + \beta^{i,m} \lambda_m + v^i$$

If the CAPM holds, then:

- The intercept of the regression (model alpha) should be zero: $\eta = 0$
    - That is, the SML goes through zero and the market return.
- The slope of the regression should be the market risk premium (mean excess return): $\lambda_m = \mathbb{E} \left[ \tilde{r}^m \right]$

This means that:

-  The Treynor Ratio, which is the slope of the SML, should be the same for all assets because they all fit on the SML.
- $R^2$ of the cross-sectional regression should be 100% (*Now you care about the $R^2$ of the regression*)
- The error term should be zero for all assets:  $v^i = 0$, $\forall i$ --> and, thus, MAE should be zero.

Note that:

- The time-series alpha is the difference between the expected return from the CAPM and the realized return for each asset (the error in the model).

In Summary:
*We want a high $R^2$, slope close to market risk premium, and zero alpha.*

That obviously does not hold in practice:
- The SML line doesn’t start at zero, $\eta > 0$.
- Slope is small, $\lambda_m$, is too small relative to the market risk premium (high beta assets are not being rewarded by additional premium).
    - *Maybe market beta is insufficient to explain all asset risk premia.*

The slope being off is a little forgiving because it equals the sample average of the factor excess returns, and the sample average is not a very precise estimate.

However, the intercept not being zero is disturbing, and probably either the t-stat is not significant or the model does not hold.

---


### 8.4. Mean Absolute Error

In this factor‐model approach, we have two different regressions—time‐series and cross‐sectional—and each produces its own notion of “error" when testing the model.

1. **Time‐Series MAE:**  
   We run a separate regression for each asset \(i\):
   \[
   R_{i,t} = \alpha_i + \beta_{i,1}\,F_{1,t} + \beta_{i,2}\,F_{2,t} + \dots + \varepsilon_{i,t}.
   \]
   - The \(\alpha_i\) is a single intercept for each asset.  
   - The code’s “TS MAE” is the average of \(\lvert \alpha_i\rvert\) across all assets:
     \[
     \text{TS MAE} = \frac{1}{N} \sum_{i=1}^N \bigl|\alpha_i\bigr|.
     \]
   - This measures how much, on average, each asset’s mean return deviates from what the time‐series factor exposures predict.

2. **Cross‐Sectional MAE:**  
   We then take each asset’s **mean** return \(\bar{R}_i\) and its estimated \(\beta_{i,k}\) from the time‐series step, and run one cross‐sectional regression:
   \[
   \bar{R}_i = \gamma_0 + \gamma_1 \,\beta_{i,1} + \gamma_2 \,\beta_{i,2} + \dots + u_i.
   \]
   - The residual \(u_i\) is how far each asset’s average return is from the fitted “factor‐pricing” line.  
   - The code’s “CS MAE” is the average of \(\lvert u_i\rvert\):
     \[
     \text{CS MAE} = \frac{1}{N} \sum_{i=1}^N \bigl|\hat{u}_i\bigr|.
     \]
   - This measures how well the estimated factor premia \(\gamma_k\) explain the *cross‐asset* variation in average returns.

So “TS MAE” is about **time‐series mispricing** (i.e., \(\alpha_i\)) for each asset, while “CS MAE” is about **cross‐sectional mispricing** (i.e., the residuals \(u_i\)) across all assets.


---
---


## 9. Time-Varying Beta

*Time-series used to scrutinize $\alpha$ over time, not $R^2$.*

We want to allow for **beta to vary over time**.

$$\tilde{r}_t^i = \alpha^i + \beta_t^{i, z} z_t + \epsilon_t^i$$

So far, we have been estimating unconditional $ \beta $:
$$\tilde{r}_t^i = \alpha^i + \beta^{i, z} z_t + \epsilon_t^i$$

Must choose a model for how $ \beta $ changes over time.

- Consider stochastic vol models above.
- Often see estimates of $ \beta_t $ using a rolling window of data (5 years?).
- Can use GARCH, other models to capture nonlinear impact.
- *Or a simpler approach:*


### 9.1. Fama-Macbeth Beta Estimation

The **Fama-Macbeth procedure** is widely used to deal with time-varying betas.

- Imposes little on the cross-sectional returns.
- Does assume no correlation across time in returns.
- Equivalent to certain GMM (generalized method of moments) specifications under these assumptions.


1. **Estimate $ \beta_t $**  
   For each security, $i$, estimate the time-series of $ \beta_t^i $. This could be done for each $t$ **using a rolling window**
      - 1 or 0.5 year for daily data (not much used for monthly data) or other methods.  
   *(If using a constant $ \beta $ just run the usual time-series regression for each security.)*

   $$\tilde{r}_t^i = \alpha^i + \beta_t^{i, z} z_t + \epsilon_t^i$$

2. **Estimate $ \lambda_t, v_t^i $**  
   For each $t$, estimate a cross-sectional regression to obtain $ \lambda_t $ and estimates of the $N$ pricing errors, $v_t^i$.

   $$\tilde{r}_t^i = \beta_t^{i, z} \underbrace{\lambda_t}_{\text{month $t$ factor premium}} + \underbrace{v_t^i}_{\text{month $t$ pricing error on asset $i$}}$$

   - *Use industry or style portfolios to test it to avoid using single names (idiosyncratic)*
   - *Last week's recording for whether to include $\alpha$.*

3. Take the average of the factor premium estimates, $ \hat{\lambda} $, and the average of the pricing errors, $ \hat{v}^i $, for every month.
---

### 9.2. Illustration of Time and Cross Regressions

Use sample means of the estimates:
$$\hat{\lambda} = \frac{1}{T} \sum_{t=1}^{T} \lambda_t, \quad \hat{v}^i = \frac{1}{T} \sum_{t=1}^{T} v_t^i$$

- *Average factor premium*
- *Standard error on factor premium*

- This allowed flexible model for $ \beta_t^{i, z} $.
- Running $ t $ cross-sectional regressions allowed $ t $ (unrelated) estimates $ \lambda_t $ and $ v_t $.

---

### 9.3. Fama-MacBeth Standard Errors

Get standard errors of the estimates by using the Law of Large Numbers for the sample means, $\hat{\lambda}$ and $\hat{v}$.

$$\text{s.e.}(\hat{\lambda}) = \frac{1}{\sqrt{T}} \sigma_\lambda$$
$$= \frac{1}{T} \sqrt{\sum_{t=1}^{T} (\lambda_t - \hat{\lambda})^2}$$

- These standard errors correct for cross-sectional correlation.
- If there is no time-series correlation in the OLS errors, then the Fama-Macbeth standard errors will equal the GMM errors.

---

### 9.4. Beyond Fama-MacBeth

The **Fama-MacBeth two-pass regression approach** is very popular to incorporate dynamic betas.

- It is easy to implement.
- It is (relatively!) easy to understand.
- It gives reasonable estimates of the standard errors.

If we want to calculate more precise standard errors, we could easily use the **Generalized Method of Moments (GMM)**.

- GMM would account for any serial correlation.
- GMM would account for the imprecision of the first-stage (time-series) estimates.


*Note:* If using full-sample time-series betas, there would be no point in using Fama-MacBeth, as this would give us the usual cross-sectional estimates.



---
---


## 10. The APT (Arbitrage Pricing Theory)

*Factor Pricing Models where the factors are chosen because they statistically work.*

Arbitrage Pricing Theory (APT) gives conditions for when a *Linear Factor Decomposition of return variation __implies__ (-->) a Linear Factor Pricing for risk premia*.

- The assumptions needed will not hold exactly.
    - In practice, we can have good LFD and bad LPM, and vice versa (ex. Momentum, which is good at pricing but terrible at hedging).
- Still, it is commonly used as a way to build LFP for risk premia in industry.

### 10.1. APT Factor Structure

Suppose we have some excess-return factors, $ \mathbf{x} $, which work well as a LFD (*Linear factor decomposition*).
    - The factors are statistically generated/chosen (not for economic reasons), because statistically they work.

$$\tilde{r}_t^i = \alpha^i + (\beta^i \cdot \mathbf{x})' \mathbf{x}_t + \epsilon_t^i$$

**APT Assumption 1:** $\Rightarrow$ *usually fails*

- *residuals are uncorrelated across regressions*:
$$\text{corr} [ \epsilon^i, \epsilon^j ] = 0, \quad i \neq j$$

That is, the factors completely describe return comovement. You can have correlation in the returns, but not in the errors.

*The problem is, if the correlations are almost zero, you cannot say much about the accuracy of the LFP. It must be zero for it to work.*

---

### 10.2. Proof of APT:

#### 10.2.1. Diversified Portfolio

Take an equally weighted portfolio of the $ n $ returns
$$\tilde{r}_t^P = \frac{1}{n} \sum_{i=1}^{n} \tilde{r}_t^i = \alpha^P + (\beta^{P, \mathbf{x}})' \mathbf{x}_t + \epsilon_t^P$$

where
$$\alpha^P = \frac{1}{n} \sum_{i=1}^{n} \alpha^i, \quad \beta^{P, \mathbf{x}} = \frac{1}{n} \sum_{i=1}^{n} \beta^{i, \mathbf{x}}, \quad \epsilon^P = \frac{1}{n} \sum_{i=1}^{n} \epsilon^i$$


#### 10.2.2. Idiosyncratic Variance

**The idiosyncratic risk of $\tilde{r}_t^P$ depends only on the residual variances.**

- By construction, the residuals are uncorrelated with the factors, $\mathbf{x}$.
- By assumption, the residuals are uncorrelated with each other.

$$\text{var} [ \epsilon^P ] = \frac{\sigma_\epsilon^2}{n} \quad \text{(no correlation, so the formula is simple)}$$

where $ \sigma_\epsilon^2 $ is the average variance of the $ n $ assets.


#### 10.2.3. Perfect Factor Structure

As the number of diversifying assets, $ n $, grows
$$\lim_{n \to \infty} \text{var} [ \epsilon^P ] = 0$$

Thus, **in the limit, $\tilde{r}_t^P$ has a perfect factor structure, with no idiosyncratic risk:**
$$\tilde{r}_t^P = \alpha^P + (\beta^{P, \mathbf{x}})' \mathbf{x}_t$$

*No idiosyncratic term.*

This says that $\tilde{r}_t^P$ can be perfectly replicated with the factors $\mathbf{x}$. If we hedge out the factor portion, we are left with alpha (constant excess return) and no risk --> *arbitrage opportunity*


#### 10.2.4. Obtaining the LFP in $ \mathbf{x} $

**APT Assumption 2:** There is no arbitrage.

Given that $\tilde{r}_t^P$ is perfectly replicated by the return factors, $\mathbf{x}$, then we must have **$\alpha^P = 0$**

Thus, taking expectations of both sides, we have a LFP:
$$\mathbb{E} [ \tilde{r}_t^P ] = (\beta^{P, \mathbf{x}})' \lambda^{\mathbf{x}}$$

where
$$\lambda^{\mathbf{x}} = \mathbb{E} [ \mathbf{x} ]$$

---

### 10.3. Explaining Variation and Pricing

The APT comes to a *stark conclusion*: **if find a perfect linear factor decomposition (LFD), then it will be a perfect linear factor pricing model (LFP)**.

It does not hold in reality because the assumptions do not hold in reality.
- We cannot find a Linear Factor Decomposition (LFD) that works so well it leaves no correlation in the residuals ($corr = 0$).
    - That is, a set of factors that explains **realized** returns across time. (*Covariation*)
- If we did, then the APT concludes the factors must also describe **expected** returns across assets. (*Risk premia*)


---

## 11. PCA (Principal Component Analysis)

By running principal component analysis on 1,000 investments, it will going to give me what the principal components are, and by definition, will have very strong explanatory power on on variation.

Thus, these components will have a very good LFD, and by the APT, they could also have a very good LFP.
- Not so useful for hedging and decomposition. 
```


