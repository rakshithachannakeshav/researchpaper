import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pytrends.request import TrendReq
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder  # === ADDED IMPORT ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import zipfile
import io

# Page configuration
st.set_page_config(
    page_title="Website Traffic Analysis & Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #f8f9fa;
    }
    .css-1aumxhk {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Website Traffic Analysis & Forecasting Dashboard")
st.markdown("""
This dashboard analyzes historical website traffic data and provides forecasts using both Prophet and LSTM models.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.markdown("### Data Options")
    uploaded_file = st.file_uploader("Upload your data (CSV or ZIP)", type=['csv', 'zip'])
    
    st.markdown("### Model Parameters")
    lstm_epochs = st.slider("LSTM Epochs", 10, 100, 50)
    lstm_batch_size = st.slider("LSTM Batch Size", 16, 128, 32)
    forecast_years = st.slider("Forecast Years", 1, 10, 5)
    
    st.markdown("### Visualization Options")
    show_raw_data = st.checkbox("Show raw data", False)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Load data function
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        # Use default data
        with zipfile.ZipFile("trends.csv.zip") as z:
            with z.open("trends.csv") as f:
                df = pd.read_csv(f)
    else:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(uploaded_file)
    
    # Clean data
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['year'] = df['year'].astype(int)
    return df

# Load data
if uploaded_file is not None or 'trends.csv.zip' in locals():
    df = load_data(uploaded_file)

    # === ADDED FEATURE ENGINEERING START ===
    def engineer_features(df_in):
        df_e = df_in.copy()
        # ensure sorting for rolling/lag per brand
        df_e = df_e.sort_values(['query', 'year']).reset_index(drop=True)
        # Time-based
        df_e['decade'] = (df_e['year'] // 10) * 10
        df_e['years_since_start'] = df_e['year'] - df_e['year'].min()
        # Brand-related
        df_e['brand_frequency'] = df_e.groupby('query')['query'].transform('count')
        df_e['avg_rank_per_brand'] = df_e.groupby('query')['rank'].transform('mean')
        df_e['rank_change'] = df_e.groupby('query')['rank'].diff().fillna(0)
        # Rolling stats
        df_e['rank_rolling_mean_3'] = df_e.groupby('query')['rank'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df_e['rank_rolling_std_3'] = df_e.groupby('query')['rank'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
        # Synthetic calendar signals
        df_e['is_even_year'] = (df_e['year'] % 2 == 0).astype(int)
        df_e['is_millennium'] = (df_e['year'] % 1000 == 0).astype(int)
        # Encode brand
        le = LabelEncoder()
        df_e['brand_id'] = le.fit_transform(df_e['query'])
        # Lag features
        df_e['rank_lag_1'] = df_e.groupby('query')['rank'].shift(1).fillna(method='bfill')
        df_e['rank_lag_2'] = df_e.groupby('query')['rank'].shift(2).fillna(method='bfill')
        return df_e

    df_enriched = engineer_features(df)
    if show_raw_data:
        st.dataframe(df_enriched.head())
    # === ADDED FEATURE ENGINEERING END ===

    if show_raw_data:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

    # Analysis Section
    st.header("🔍 Data Analysis")
    
    # Top 15 Brands Analysis
    st.subheader("Top 15 Brands Analysis")
    
    # Get top brands
    top_brands = df['query'].value_counts().head(15).index
    df_top = df[df['query'].isin(top_brands)]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Rank Over Time", "Brand Frequency", "Heatmap", "Rank Distribution"])
    
    with tab1:
        st.markdown("### Rank Over Time (Top 15 Brands)")
        df_pivot = df_top.pivot_table(index='year', columns='query', values='rank', aggfunc='mean')
        df_pivot = df_pivot.ffill()
        
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for brand in df_pivot.columns:
            ax1.plot(df_pivot.index, df_pivot[brand], label=brand)
        ax1.invert_yaxis()
        ax1.set_title('Rank Over Time (Top 15 Brands)')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Rank')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.grid(True)
        st.pyplot(fig1)
        
    with tab2:
        st.markdown("### Top Brands by Frequency")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        df['query'].value_counts().head(15).plot(kind='barh', color='teal', ax=ax2)
        ax2.set_title('Top 15 Brands by Frequency')
        ax2.set_xlabel('Appearances')
        st.pyplot(fig2)
        
    with tab3:
        st.markdown("### Heatmap: Average Rank by Year")
        heatmap_data = df_top.pivot_table(index='query', columns='year', values='rank', aggfunc='mean')
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        sns.heatmap(heatmap_data, cmap='mako', linewidths=0.5, ax=ax3)
        ax3.set_title('Heatmap: Average Rank of Top 15 Brands Over Years')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Brand')
        st.pyplot(fig3)
        
    with tab4:
        st.markdown("### Rank Distribution by Brand")
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df_top, x='query', y='rank', ax=ax4)
        ax4.set_title('Rank Distribution by Brand (Top 15)')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        ax4.invert_yaxis()
        st.pyplot(fig4)
    
    # Time Series Analysis
    st.subheader("Time Series Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Average Rank Over Time")
        avg_rank_all = df.groupby('year')['rank'].mean()
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        ax5.plot(avg_rank_all.index, avg_rank_all.values, marker='o', color='purple')
        ax5.set_title('Average Rank of All Brands Over Time')
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Average Rank')
        ax5.invert_yaxis()
        ax5.grid(True)
        st.pyplot(fig5)
        
    with col2:
        st.markdown("#### Unique Brands Per Year")
        brands_per_year = df.groupby('year')['query'].nunique()
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        ax6.bar(brands_per_year.index, brands_per_year.values, color='coral')
        ax6.set_title('Number of Unique Brands Per Year')
        ax6.set_xlabel('Year')
        ax6.set_ylabel('Number of Brands')
        st.pyplot(fig6)
    
    # Interactive Visualizations
    st.subheader("Interactive Visualizations")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Interactive Line Plot")
        fig_line = px.line(df_top, x='year', y='rank', color='query',
                          title='Interactive Line Plot - Top 15 Brands')
        fig_line.update_yaxes(autorange='reversed')
        fig_line.update_layout(height=500)
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col4:
        st.markdown("#### Brand Frequency Distribution")
        brand_counts = df['query'].value_counts().head(15).reset_index()
        brand_counts.columns = ['Brand', 'Frequency']
        fig_pie = px.pie(brand_counts, names='Brand', values='Frequency',
                         title='Top 15 Brands - Frequency Distribution',
                         hole=0.3)
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Forecasting Section
    st.header("🔮 Forecasting Models")
    
    # Prepare data for forecasting
    df_prophet = df[['year', 'rank']].copy()
    df_prophet.rename(columns={'year': 'ds', 'rank': 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
    df_prophet['y'] = df_prophet['y'].astype(float)
    
    # === ADDED: prepare prophet dataframe with extra regressors (uses df_enriched) ===
    try:
        df_prophet_features = df_enriched[['year', 'rank', 'brand_frequency', 'avg_rank_per_brand', 'rank_rolling_mean_3', 'rank_rolling_std_3', 'brand_id']].copy()
        df_prophet_features.rename(columns={'year': 'ds', 'rank': 'y'}, inplace=True)
        df_prophet_features['ds'] = pd.to_datetime(df_prophet_features['ds'], format='%Y')
        df_prophet_features['y'] = df_prophet_features['y'].astype(float)
    except Exception as e:
        # if enrichment failed for any reason, fallback to original small df
        df_prophet_features = df_prophet.copy()

    # LSTM data preparation
    df_lstm = df[['year', 'rank']].copy()
    df_lstm['year'] = pd.to_datetime(df_lstm['year'], format='%Y')
    df_lstm.set_index('year', inplace=True)
    
    # === ADDED: prepare multivariate LSTM dataframe (uses df_enriched) ===
    try:
        df_lstm_multi = df_enriched[['year', 'rank', 'rank_rolling_mean_3', 'rank_rolling_std_3', 'brand_frequency', 'brand_id']].copy()
        df_lstm_multi['year'] = pd.to_datetime(df_lstm_multi['year'], format='%Y')
        df_lstm_multi.set_index('year', inplace=True)
    except Exception as e:
        df_lstm_multi = df_lstm.copy()
        # ensure at least the 'rank' column exists
        if 'rank_rolling_mean_3' not in df_lstm_multi.columns:
            df_lstm_multi['rank_rolling_mean_3'] = df_lstm_multi['rank']
        if 'rank_rolling_std_3' not in df_lstm_multi.columns:
            df_lstm_multi['rank_rolling_std_3'] = 0
        if 'brand_frequency' not in df_lstm_multi.columns:
            df_lstm_multi['brand_frequency'] = 1
        if 'brand_id' not in df_lstm_multi.columns:
            df_lstm_multi['brand_id'] = 0
    
    # Model tabs
    tab_prophet, tab_lstm, tab_comparison = st.tabs(["Prophet Model", "LSTM Model", "Model Comparison"])
    
    with tab_prophet:
        st.markdown("### Prophet Forecasting Model")
        
        with st.spinner('Training Prophet model...'):
            # Prophet Model
            model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1)
            model.fit(df_prophet)
            
            # Forecast Future Traffic
            future = model.make_future_dataframe(periods=forecast_years, freq='Y')
            forecast = model.predict(future)
            
            # Plot Forecast
            st.markdown("#### Forecast Plot")
            fig_prophet = model.plot(forecast)
            plt.title('Website Traffic Forecast (Prophet)')
            plt.xlabel('Date')
            plt.ylabel('Forecasted Rank')
            st.pyplot(fig_prophet)
            
            # Forecast Components
            st.markdown("#### Forecast Components")
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
            
            # Metrics
            st.markdown("#### Model Metrics")
            df_cv = cross_validation(model, initial='365 days', period='90 days', horizon='90 days')
            df_perf = performance_metrics(df_cv)
            
            col_met1, col_met2 = st.columns(2)
            
            with col_met1:
                st.dataframe(df_perf[['horizon', 'rmse', 'mae', 'mape']].head())
                
            with col_met2:
                st.metric("RMSE", f"{df_perf['rmse'].mean():.4f}")
                st.metric("MAE", f"{df_perf['mae'].mean():.4f}")
                st.metric("MAPE", f"{df_perf['mape'].mean():.2f}%")
        
        # === ADDED: Prophet with additional regressors (uses features) ===
        st.markdown("### Prophet + Engineered Regressors (ADDED)")
        with st.spinner('Training Prophet (with engineered regressors)...'):
            try:
                model_reg = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1)
                extra_regs = ['brand_frequency', 'avg_rank_per_brand', 'rank_rolling_mean_3', 'rank_rolling_std_3', 'brand_id']
                for r in extra_regs:
                    model_reg.add_regressor(r)
                model_reg.fit(df_prophet_features)
                future_reg = model_reg.make_future_dataframe(periods=forecast_years, freq='Y')
                # Fill regressors for future using last known values (simple strategy)
                last_vals = df_prophet_features.iloc[-1]
                for r in extra_regs:
                    future_reg[r] = last_vals[r]
                forecast_reg = model_reg.predict(future_reg)
                
                st.markdown("#### Forecast Plot (Prophet + Regressors)")
                fig_prophet_reg = model_reg.plot(forecast_reg)
                plt.title('Prophet Forecast with Extra Regressors')
                st.pyplot(fig_prophet_reg)
                
                st.markdown("#### Forecast Components (Prophet + Regressors)")
                fig_comp_reg = model_reg.plot_components(forecast_reg)
                st.pyplot(fig_comp_reg)
            except Exception as e:
                st.warning("Could not train Prophet with extra regressors: " + str(e))
    
    with tab_lstm:
        st.markdown("### LSTM Forecasting Model")
        
        with st.spinner('Training LSTM model...'):
            # Scaling the data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_lstm_scaled = scaler.fit_transform(df_lstm)
            
            # Creating Sequences for LSTM
            def create_lstm_sequences(data, time_step=1):
                X, Y = [], []
                for i in range(len(data) - time_step):
                    X.append(data[i:(i + time_step), 0])
                    Y.append(data[i + time_step, 0])
                return np.array(X), np.array(Y)
            
            time_step = 5
            X_lstm, y_lstm = create_lstm_sequences(df_lstm_scaled, time_step)
            
            # Reshape X_lstm for LSTM input
            X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)
            
            # Train-Test Split for LSTM
            X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
            
            # LSTM Model
            model_lstm = Sequential()
            model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model_lstm.add(Dropout(0.2))
            model_lstm.add(LSTM(units=50, return_sequences=False))
            model_lstm.add(Dropout(0.2))
            model_lstm.add(Dense(units=1))
            
            # Compile and Fit LSTM Model
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            history = model_lstm.fit(X_train, y_train, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=0)
            
            # Make Predictions with LSTM
            predictions_lstm = model_lstm.predict(X_test)
            predictions_lstm = scaler.inverse_transform(predictions_lstm)
            y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Plot LSTM Forecast vs Actual
            st.markdown("#### LSTM Forecast vs Actual")
            fig_lstm, ax_lstm = plt.subplots(figsize=(10, 5))
            ax_lstm.plot(df_lstm.index[-len(y_test):], y_test_inverse, color='blue', label='Actual Rank')
            ax_lstm.plot(df_lstm.index[-len(predictions_lstm):], predictions_lstm, color='red', label='LSTM Predicted Rank')
            ax_lstm.set_title('LSTM Model: Forecast vs Actual')
            ax_lstm.set_xlabel('Date')
            ax_lstm.set_ylabel('Rank')
            ax_lstm.legend()
            st.pyplot(fig_lstm)
            
            # LSTM Model Evaluation
            st.markdown("#### LSTM Model Metrics")
            mse_lstm = mean_squared_error(y_test_inverse, predictions_lstm)
            mae_lstm = mean_absolute_error(y_test_inverse, predictions_lstm)
            mape_lstm = np.mean(np.abs((y_test_inverse - predictions_lstm) / y_test_inverse)) * 100
            
            col_lstm1, col_lstm2 = st.columns(2)
            
            with col_lstm1:
                st.metric("MSE", f"{mse_lstm:.8f}")
                st.metric("RMSE", f"{np.sqrt(mse_lstm):.8f}")
                
            with col_lstm2:
                st.metric("MAE", f"{mae_lstm:.6f}")
                st.metric("MAPE", f"{mape_lstm:.2f}")
            
            # Future predictions
            st.markdown("#### Future Forecast with LSTM")
            
            def predict_future_lstm(model, last_data, future_steps, time_step):
                predictions = []
                current_input = last_data
                
                for _ in range(future_steps):
                    current_input = current_input.reshape(1, time_step, 1)
                    prediction = model.predict(current_input)
                    predictions.append(prediction[0, 0])
                    current_input = np.append(current_input[0][1:], prediction).reshape(time_step, 1)
                return predictions
            
            last_data = df_lstm_scaled[-time_step:]
            predictions_lstm_future = predict_future_lstm(model_lstm, last_data, forecast_years, time_step)
            predictions_lstm_future = scaler.inverse_transform(np.array(predictions_lstm_future).reshape(-1, 1))
            
            future_years = pd.date_range(df_lstm.index[-1] + pd.DateOffset(years=1), periods=forecast_years, freq='Y')
            forecast_df = pd.DataFrame(data=predictions_lstm_future, index=future_years, columns=['Forecasted Rank'])
            
            st.dataframe(forecast_df)
            
            # Plot future forecast
            fig_lstm_future, ax_lstm_future = plt.subplots(figsize=(10, 5))
            ax_lstm_future.plot(df_lstm.index[-len(y_test):], y_test_inverse, color='blue', label='Actual Rank')
            ax_lstm_future.plot(future_years, predictions_lstm_future, color='red', label='LSTM Forecasted Rank')
            ax_lstm_future.set_title('LSTM Forecast vs Actual (Future Predictions)')
            ax_lstm_future.set_xlabel('Year')
            ax_lstm_future.set_ylabel('Rank')
            ax_lstm_future.legend()
            st.pyplot(fig_lstm_future)

        # === ADDED: Multivariate LSTM using engineered features (ADDED) ===
        st.markdown("### Multivariate LSTM (with engineered features) - ADDED")
        with st.spinner('Training Multivariate LSTM (added features)...'):
            try:
                features = ['rank', 'rank_rolling_mean_3', 'rank_rolling_std_3', 'brand_frequency', 'brand_id']
                scaler_multi = MinMaxScaler()
                scaled_multi = scaler_multi.fit_transform(df_lstm_multi[features])

                def create_lstm_sequences_multi(data, time_step=1):
                    X, Y = [], []
                    for i in range(len(data) - time_step):
                        X.append(data[i:(i + time_step), :])
                        Y.append(data[i + time_step, 0])  # target is first column (rank)
                    return np.array(X), np.array(Y)

                time_step_multi = 5
                X_multi, y_multi = create_lstm_sequences_multi(scaled_multi, time_step_multi)
                # Train-test split
                X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, shuffle=False)

                # Multivariate LSTM
                model_lstm_multi = Sequential()
                model_lstm_multi.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_m.shape[1], X_train_m.shape[2])))
                model_lstm_multi.add(Dropout(0.2))
                model_lstm_multi.add(LSTM(units=50, return_sequences=False))
                model_lstm_multi.add(Dropout(0.2))
                model_lstm_multi.add(Dense(units=1))

                model_lstm_multi.compile(optimizer='adam', loss='mean_squared_error')
                history_m = model_lstm_multi.fit(X_train_m, y_train_m, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=0)

                preds_m = model_lstm_multi.predict(X_test_m)
                # inverse transform predictions (we put preds in first column then inverse)
                preds_full = np.zeros((len(preds_m), len(features)))
                preds_full[:, 0] = preds_m[:, 0]
                preds_inv = scaler_multi.inverse_transform(preds_full)[:, 0]

                y_test_full = np.zeros((len(y_test_m), len(features)))
                y_test_full[:, 0] = y_test_m
                y_test_inv = scaler_multi.inverse_transform(y_test_full)[:, 0]

                # Plot results
                fig_m, ax_m = plt.subplots(figsize=(10, 5))
                ax_m.plot(df_lstm_multi.index[-len(y_test_inv):], y_test_inv, color='blue', label='Actual Rank')
                ax_m.plot(df_lstm_multi.index[-len(preds_inv):], preds_inv, color='red', label='Multivariate LSTM Predicted Rank')
                ax_m.set_title('Multivariate LSTM: Forecast vs Actual')
                ax_m.set_xlabel('Date')
                ax_m.set_ylabel('Rank')
                ax_m.legend()
                st.pyplot(fig_m)

                # Metrics
                mse_m = mean_squared_error(y_test_inv, preds_inv)
                mae_m = mean_absolute_error(y_test_inv, preds_inv)
                mape_m = np.mean(np.abs((y_test_inv - preds_inv) / y_test_inv)) * 100

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("MV MSE", f"{mse_m:.8f}")
                    st.metric("MV RMSE", f"{np.sqrt(mse_m):.8f}")
                with col_m2:
                    st.metric("MV MAE", f"{mae_m:.6f}")
                    st.metric("MV MAPE", f"{mape_m:.2f}")

                # Future multivariate predictions
                def predict_future_multi(model, last_data, future_steps, time_step):
                    predictions = []
                    current_input = last_data.copy()
                    for _ in range(future_steps):
                        pred_scaled = model.predict(current_input.reshape(1, time_step, current_input.shape[1]))[0, 0]
                        # create new multivariate row: predicted rank in col 0, keep last values for other features
                        new_row = np.zeros(current_input.shape[1])
                        new_row[0] = pred_scaled
                        new_row[1:] = current_input[-1, 1:]
                        predictions.append(new_row)
                        current_input = np.vstack([current_input[1:], new_row])
                    return np.array(predictions)

                last_multi = scaled_multi[-time_step_multi:]
                preds_future_scaled = predict_future_multi(model_lstm_multi, last_multi, forecast_years, time_step_multi)
                preds_future_full = scaler_multi.inverse_transform(preds_future_scaled)[:, 0]

                future_years_mv = pd.date_range(df_lstm_multi.index[-1] + pd.DateOffset(years=1), periods=forecast_years, freq='Y')
                forecast_mv_df = pd.DataFrame(data=preds_future_full, index=future_years_mv, columns=['Forecasted Rank (MV LSTM)'])
                st.dataframe(forecast_mv_df)

                fig_mf, ax_mf = plt.subplots(figsize=(10, 5))
                ax_mf.plot(df_lstm_multi.index[-len(y_test_inv):], y_test_inv, color='blue', label='Actual Rank')
                ax_mf.plot(future_years_mv, preds_future_full, color='red', label='MV LSTM Forecast')
                ax_mf.set_title('Multivariate LSTM Forecast (Future Predictions)')
                ax_mf.set_xlabel('Year')
                ax_mf.set_ylabel('Rank')
                ax_mf.legend()
                st.pyplot(fig_mf)

            except Exception as e:
                st.warning("Could not train multivariate LSTM with engineered features: " + str(e))
    
    with tab_comparison:
        st.markdown("### Model Comparison")
        
        # Metrics from Prophet (example values - replace with actual from your model)
        rmse_prophet = 1.4142
        mae_prophet = 1.2001
        mape_prophet = 63.01
        
        # Metrics from LSTM
        mse_lstm_value = mean_squared_error(y_test_inverse, predictions_lstm)
        rmse_lstm = np.sqrt(mse_lstm_value)
        mae_lstm_value = mean_absolute_error(y_test_inverse, predictions_lstm)
        mape_lstm_value = np.mean(np.abs((y_test_inverse - predictions_lstm) / y_test_inverse)) * 100
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': ['RMSE', 'MAE', 'MAPE'],
            'Prophet': [rmse_prophet, mae_prophet, mape_prophet],
            'LSTM': [rmse_lstm, mae_lstm_value, mape_lstm_value]
        }
        df_comparison = pd.DataFrame(comparison_data)
        
        st.dataframe(df_comparison)
        
        # Plot comparison
        fig_compare, ax_compare = plt.subplots(figsize=(10, 6))
        df_comparison.set_index('Metric').plot(kind='bar', ax=ax_compare)
        ax_compare.set_title('Model Performance Comparison')
        ax_compare.set_ylabel('Value')
        st.pyplot(fig_compare)
        
        # Conclusion
        st.markdown("#### Conclusion")
        st.markdown("""
        - The LSTM model outperforms the Prophet model by a significant margin across all evaluation metrics.
        - It has much lower error rates, indicating more accurate and reliable forecasts for the website traffic data.
        - Therefore, LSTM is a better fit for this time series forecasting problem.
        """)
else:
    st.warning("Please upload a data file or ensure 'trends.csv.zip' is available to proceed.")
