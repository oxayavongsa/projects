<h1>Prediction of Life Expectancy based on Gender</h1>

<p>This project aims to predict life expectancy based on the US state and gender using machine learning models. The project involves data cleaning, exploratory data analysis, feature engineering, model training, and evaluation.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#data-description">Data Description</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#model-training-and-evaluation">Model Training and Evaluation</a></li>
    <li><a href="#prediction">Prediction</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#license">License</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
<p>This project predicts life expectancy based on US states and gender using datasets from the years 2010-2015 and 2020. The models used include Linear Regression and Random Forest Regressor. The project also includes hyperparameter tuning using GridSearchCV to improve the Random Forest model's performance.</p>

<h2 id="data-description">Data Description</h2>
<h3>Datasets</h3>
<ol>
    <li>
        <strong>U.S. Life Expectancy at Birth by State and Census Tract (2010-2015)</strong>
        <ul>
            <li>Columns: <code>State</code>, <code>County</code>, <code>Census Tract Number</code>, <code>Life Expectancy</code>, <code>Life Expectancy Range</code>, <code>Life Expectancy Standard Error</code></li>
        </ul>
    </li>
    <li>
        <strong>U.S. State Life Expectancy by Sex (2020)</strong>
        <ul>
            <li>Columns: <code>State</code>, <code>Sex</code>, <code>LE</code> (Life Expectancy), <code>SE</code> (Standard Error), <code>Quartile</code></li>
        </ul>
    </li>
</ol>

<h3>Data Cleaning</h3>
<ul>
    <li>Unnecessary columns were dropped.</li>
    <li>Missing values were handled by filling with the mean of respective columns.</li>
    <li>Categorical variables were encoded for model training.</li>
</ul>

<h2 id="installation">Installation</h2>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/yourusername/life-expectancy-prediction.git
cd life-expectancy-prediction</code></pre>
    </li>
    <li>Install the required packages:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
</ol>

<h2 id="usage">Usage</h2>
<ol>
    <li><strong>Load and Inspect the Data</strong>
        <pre><code>df = pd.read_csv('U.S._Life_Expectancy_at_Birth_by_State_and_Census_Tract_-_2010-2015.csv')
df2 = pd.read_csv('U.S._State_Life_Expectancy_by_Sex__2020.csv')</code></pre>
    </li>
    <li><strong>Data Cleaning</strong>
        <pre><code>columns_to_drop = ['Census Tract Number', 'State', 'County']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
df2 = df2.drop(columns=['Quartile'])
df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
df2 = df2.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
df2['Sex'] = df2['Sex'].map({'Male': 0, 'Female': 1, 'Total': 2})</code></pre>
    </li>
    <li><strong>Exploratory Data Analysis</strong>
        <pre><code>sns.boxplot(x='Sex', y='LE', data=df2)
sns.histplot(df2['LE'], kde=True, bins=30)
sns.violinplot(x='State', y='LE', hue='Sex', data=df2)</code></pre>
    </li>
    <li><strong>Model Training and Evaluation</strong>
        <pre><code>from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = pd.get_dummies(df2[['State', 'Sex']], columns=['State'])
y = df2['LE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred)}")
print(f"Linear Regression R2: {r2_score(y_test, y_pred)}")

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred_rf)}")
print(f"Random Forest R2: {r2_score(y_test, y_pred_rf)}")</code></pre>
    </li>
    <li><strong>Hyperparameter Tuning with Grid Search</strong>
        <pre><code>from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, verbose=2)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)
print(f"Best Random Forest MSE: {mean_squared_error(y_test, y_pred_best_rf)}")
print(f"Best Random Forest R2: {r2_score(y_test, y_pred_best_rf)}")</code></pre>
    </li>
</ol>

<h2 id="prediction">Prediction</h2>

<h3>Predict Life Expectancy</h3>
<p>Use the <code>predict_life_expectancy</code> function to predict life expectancy based on state and gender.</p>
<pre><code>def predict_life_expectancy(state, gender):
    gender_code = {'Male': 0, 'Female': 1, 'Other': 2}[gender]
    input_data = pd.DataFrame({'Sex': [gender_code], 'State_' + state: [1]})
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]
    prediction = best_rf_model.predict(input_data)
    return prediction[0]</code></pre>

<h3>Example</h3>
<pre><code>print(predict_life_expectancy('California', 'Female'))</code></pre>

<h2 id="results">Results</h2>

<p>The Random Forest model with hyperparameter tuning achieved better performance:</p>
<ul>
    <li><strong>Linear Regression</strong>
        <ul>
            <li>Mean Squared Error: 10.07</li>
            <li>R-squared: 0.11</li>
        </ul>
    </li>
    <li><strong>Random Forest</strong>
        <ul>
            <li>Mean Squared Error: 5.79</li>
            <li>R-squared: 0.49</li>
        </ul>
    </li>
    <li><strong>Best Random Forest (after hyperparameter tuning)</strong>
        <ul>
            <li>Mean Squared Error: 5.91</li>
            <li>R-squared: 0.48</li>
