import streamlit as st
import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_rows', 100)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.cluster import KMeans


data = pd.read_csv('data.csv', encoding='latin-1')


st.header("Project Forecasting with Streamlit :green[Python]")
st.subheader(":red[BD FGA ITI 2024]",divider='gray')
st.subheader(
  '''
  Anggota tim:
  
  ~Rahmat Guntur Husodo

  ~Dandy Darmawan Al Yahmin
  
  ~Bahrin Rohimul Umna
  
  ~Muhammad Bimo Anggoro Seno


  '''
,divider='gray')


#Project Understanding
st.header(":green[Project Understanding:]")
st.subheader(":blue[1. Bussiness Understanding]")
st.write('''
  Perusahaan ritel daring non-toko yang terdaftar dan berbasis di Inggris Raya ini beroperasi dalam penjualan hadiah unik untuk berbagai acara. Kumpulan data transnasional ini mencakup semua transaksi yang terjadi antara 01 Desember 2010 dan 09 Desember 2011. Data ini mencerminkan aktivitas bisnis perusahaan selama satu tahun penuh, menawarkan wawasan berharga tentang perilaku pembelian pelanggan dan tren penjualan.
''')
st.subheader(":blue[2. Data Understanding:]")
st.write('''
  Data ini mencakup penjualan dari pengecer daring di Inggris. Dengan mempertimbangkan bahwa biaya penyimpanan dapat tinggi dan pengiriman tepat waktu sangat penting untuk bersaing, analisis ini bertujuan untuk membantu pengecer dengan memperkirakan jumlah produk yang terjual setiap hari. Pemahaman ini akan memungkinkan pengecer untuk mengelola stok secara lebih efisien dan meningkatkan kepuasan pelanggan melalui pengiriman yang tepat waktu.
''')
st.subheader("Project Understanding finish!",divider='gray')

#Data Exploration
st.header('''
  :green[Data Exploration:]
''')
st.write("Informasi Data:")
n_rows, n_cols = data.shape
dtypes = data.dtypes
missing_values = data.isnull().sum()
non_missing_values = data.notnull().sum()
summary_df = pd.DataFrame({
    'Column': data.columns,
    'Data Type': dtypes,
    'Non-Null Count': non_missing_values,
    'Null Count': missing_values
})
data["InvoiceDate"] = pd.to_datetime(data.InvoiceDate, cache=True)
st.write("Datafile starts with timepoint {}".format(data.InvoiceDate.min()))
st.write("Datafile ends with timepoint {}".format(data.InvoiceDate.max()))
st.write("Datafile period was {}".format((data.InvoiceDate.max() - data.InvoiceDate.min())))
st.write(summary_df)
st.write("Contoh Data:")
st.write(data.head(10))
st.write("Statistic Data:")
st.write(data.describe())
st.subheader("Data Exploration finish!",divider='gray')


#Data Cleaning
unit_price_mean = data['UnitPrice'].mean()
data['UnitPrice'] = data['UnitPrice'].replace(0, unit_price_mean)
data['Description'].fillna('Deskripsi tidak tersedia')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

#Data Analytics
st.header('''
  :green[Data Analytics:]
''')
st.write(":blue[1. Top 10 Produk dengan jumlah jual terbanyak:]")
sales_by_product = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=sales_by_product.values, y=sales_by_product.index, palette='viridis')
plt.xlabel('Quantity Sold')
plt.ylabel('Product')
plt.title('Top 10 Products by Sales')
st.pyplot(plt)
st.write('''
  1. Produk Terlaris:
  "WORLD WAR 2 GLIDERS ASSTD DESIGNS" dan "JUMBO BAG RED RETROSPOT" adalah dua produk terlaris, menunjukkan permintaan tinggi untuk barang-barang ini.

  2. Volume Penjualan Signifikan:
  Produk-produk ini telah terjual dalam jumlah yang jauh lebih besar dibandingkan dengan produk lainnya, mengindikasikan popularitas yang tinggi di kalangan pelanggan.

  3. Ragam Produk:
  Daftar ini mencakup berbagai macam produk seperti ornamen, wadah, lampu malam, dan berbagai barang bertema, yang mencerminkan portofolio produk yang luas.

  4. Kategori Populer:
  Barang-barang yang berhubungan dengan desain retro dan tema unik (misalnya, "PACK OF 72 RETROSPOT CAKE CASES") tampaknya memiliki performa baik, yang mungkin menunjukkan preferensi pelanggan terhadap barang-barang vintage atau unik.

  5. Implikasi Persediaan:
  Mengingat volume penjualan yang tinggi, disarankan untuk menjaga persediaan yang cukup untuk produk-produk yang berkinerja baik ini untuk memenuhi permintaan pelanggan dan menghindari kehabisan stok.
''')
st.write(":blue[2. Top 10 produk dengan keuntungan terbanyak]")
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
top_revenue_products = data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index, palette='viridis')
plt.xlabel('Total Revenue')
plt.ylabel('Product')
plt.title('Top 10 Products by Revenue')
st.pyplot(plt)
st.write('''
  1. Produk dengan Pendapatan Tertinggi:
  DOTCOM POSTAGE adalah produk dengan pendapatan tertinggi, dengan total pendapatan yang jauh melampaui produk lainnya. REGENCY CAKESTAND 3 TIER berada di urutan kedua dengan pendapatan yang juga signifikan.

  2. Produk Lainnya dengan Pendapatan Tinggi:
  Produk seperti WHITE HANGING HEART T-LIGHT HOLDER, PARTY BUNTING, dan JUMBO BAG RED RETROSPOT termasuk dalam kategori produk dengan pendapatan tinggi, meskipun tidak setinggi dua produk teratas.

  3. Ragam Produk yang Menghasilkan Pendapatan:
  Produk dengan berbagai macam jenis, mulai dari dekorasi rumah (seperti WHITE HANGING HEART T-LIGHT HOLDER dan PARTY BUNTING), peralatan pesta (seperti PAPER CHAIN KIT 50'S CHRISTMAS), hingga barang kebutuhan khusus (seperti RABBIT NIGHT LIGHT), semuanya menyumbang pendapatan yang signifikan.

  4. Distribusi Pendapatan:
  Pendapatan cenderung terdistribusi dengan baik di antara berbagai jenis produk, menunjukkan diversifikasi yang baik dalam portofolio produk.

  5. Peluang Peningkatan:
  Produk dengan pendapatan lebih rendah dalam daftar top 10, seperti ASSORTED COLOUR BIRD ORNAMENT dan CHILLI LIGHTS, mungkin memiliki potensi untuk ditingkatkan baik dari segi pemasaran maupun strategi penjualan.
''')
st.write(":blue[3. Pendapatan Bulanan dalam 1 Tahun:]")
data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')
monthly_sales = data.groupby('YearMonth').size()
plt.figure(figsize=(12, 6))
monthly_sales.plot(marker='o')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Transaksi')
plt.title('Penjualan berdasarkan Bulan dalam Setahun:')
plt.grid(True)
st.pyplot(plt)
st.write('''
  1. Peningkatan Signifikan di Bulan November:
  Terjadi lonjakan penjualan yang sangat signifikan pada bulan November, dengan jumlah transaksi mencapai puncaknya. Ini kemungkinan disebabkan oleh musim belanja akhir tahun, termasuk penjualan Black Friday dan persiapan untuk liburan Natal.

  2. Penurunan Drastis di Bulan Desember:
  Setelah puncak penjualan di bulan November, jumlah transaksi turun drastis di bulan Desember. Hal ini mungkin disebabkan oleh berakhirnya musim belanja liburan dan persediaan stok yang sudah dibeli di bulan sebelumnya.

  3. Stabilitas Penjualan di Bulan Lainnya:
  Sepanjang tahun, jumlah transaksi relatif stabil dengan beberapa fluktuasi kecil. Tidak ada lonjakan signifikan selain di bulan November.

  4. Tren Musiman:
  Data menunjukkan adanya tren musiman yang kuat, terutama terkait dengan belanja akhir tahun. Ini bisa dijadikan acuan untuk strategi pemasaran dan persiapan stok di masa mendatang.

  5. Strategi Bisnis:
  Untuk memaksimalkan pendapatan, perusahaan bisa memfokuskan strategi pemasaran dan promosi pada periode sebelum November. Menyiapkan stok yang cukup untuk menghadapi lonjakan permintaan juga menjadi kunci penting.
''')
st.write(":blue[4. Pendapatan Harian rata-rata Perminggu]")
data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()
weekday_sales = data.groupby('DayOfWeek').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.figure(figsize=(10, 6))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values, palette='pastel')
plt.xlabel('Hari')
plt.ylabel('Jumlah Transaksi')
plt.title('Penjualan berdasarkan Hari dalam Seminggu')
st.pyplot(plt)
st.write('''
  1. Hari dengan Penjualan Tertinggi:
  Penjualan tertinggi terjadi pada hari Selasa dan Kamis. Kedua hari ini memiliki jumlah transaksi yang hampir sama, dan keduanya lebih tinggi dibandingkan hari-hari lainnya.

  2. Hari dengan Penjualan Terendah:
  Penjualan terendah terjadi pada hari Minggu. Ini menunjukkan bahwa aktivitas penjualan berkurang signifikan pada hari tersebut.

  3. Distribusi Penjualan di Hari Kerja:
  Secara umum, penjualan cenderung tinggi dari Senin hingga Kamis, dengan sedikit penurunan pada hari Jumat.

  4. Penurunan di Akhir Pekan:
  Terdapat penurunan yang jelas dalam jumlah transaksi pada akhir pekan (Sabtu dan Minggu) dibandingkan dengan hari kerja.
''')
st.write(":blue[5. Order yang di cancel:]")
cancelled = pd.Series(np.where(data.InvoiceNo.apply(lambda l: l[0]=="C"), 'Canceled', 'Not Canceled'))
fig = px.pie(values=cancelled.value_counts(), names=cancelled.value_counts().index)
st.write(fig)
st.write('''
  Karena tidak ada cara untuk memeriksa mengapa pesanan tersebut dibatalkan dan mempengaruhi data karena Kuantitas dan Harga Satuan negatifnya, saya yakin lebih baik untuk menghapusnya bersama outlier
''')
code = '''
def remove_outliers(column):
    Q1 = column.quantile(.25)
    Q3 = column.quantile(.75)
    IQR = Q3 - Q1
    column = column[((Q1 - 1.5 * IQR) <= column) & (column  <= (Q3 + 1.5 * IQR))]
    return column

data.Quantity = remove_outliers(data.Quantity)
data.UnitPrice = remove_outliers(data.UnitPrice)
data.dropna(subset=['Quantity','UnitPrice'],inplace=True)
data['Revenue'] = data.Quantity * data.UnitPrice
data = data[data.Revenue>0.1]
'''
st.code(code)
def remove_outliers(column):
    Q1 = column.quantile(.25)
    Q3 = column.quantile(.75)
    IQR = Q3 - Q1
    column = column[((Q1 - 1.5 * IQR) <= column) & (column  <= (Q3 + 1.5 * IQR))]
    return column

data.Quantity = remove_outliers(data.Quantity)
data.UnitPrice = remove_outliers(data.UnitPrice)
data.dropna(subset=['Quantity','UnitPrice'],inplace=True)
data['Revenue'] = data.Quantity * data.UnitPrice
data = data[data.Revenue>0.1]
cancelled = pd.Series(np.where(data.InvoiceNo.apply(lambda l: l[0]=="C"), 'Cancelled', 'Not Cancelled'))
fig = px.pie(values=cancelled.value_counts(), names=cancelled.value_counts().index)
st.write(fig)
st.write(":blue[6. Pendapatan dari waktu ke waktu]")
data['InvoiceDate'] = data['InvoiceDate'].dt.date
ddate = data[['InvoiceDate','Revenue']].groupby('InvoiceDate').sum()
revenue_moving_average = ddate.rolling(
    window=150,       
    center=True,      
    min_periods=75,  
).mean()
fig = make_subplots(rows=1, cols=1, vertical_spacing=0.08)
fig.add_trace(go.Scatter(x=ddate.index, y=ddate.iloc[:,0],marker=dict(color= '#60d92b'), name='Moving Average'))
fig.add_trace(go.Scatter(x=revenue_moving_average.index,y=revenue_moving_average.iloc[:,0],mode='lines',name='Trend', marker=dict(color= '#347d48')))
st.write(fig)
st.write(":blue[7. Negara dengan jumlah order terbanyak: ]")
fig = px.pie(values=data['Country'].value_counts().head(10), names=data['Country'].value_counts().head(10).index, title='Countries')
st.write(fig)
st.write(":blue[8. Negara dengan jumlah pembelian terbesar:]")
crdata = data[['Country', 'Revenue']].groupby('Country').mean().sort_values(by='Revenue', ascending=False)
fig = px.pie(crdata, names=crdata.index, values='Revenue', title='Revenue by Country')
fig.update_layout(autosize=False, width=950, height=600, title_text="Revenue Distribution")
st.plotly_chart(fig)
st.subheader("Data Analytics finish!",divider='gray')


#Forecasting
st.header(":green[Forecasting:]")
data['monthday'] = pd.to_datetime(data['InvoiceDate']).dt.day
ddate = data[['monthday','Revenue']].groupby('monthday').sum()
data['weakday'] = pd.to_datetime(data['InvoiceDate']).dt.dayofweek
ddate = data[['weakday','Revenue']].groupby('weakday').sum()
crdata = data[['Country','Revenue']].groupby('Country').mean().sort_values(by = 'Revenue', ascending = False)
isnull = data.isnull().sum().sort_values(ascending=False).to_frame()
isnull.columns = ['How_many']
isnull['precentage'] = np.around(((isnull / len(data) * 100)[(isnull / len(data) * 100) != 0]), decimals=2)
data.loc[data.Description.isnull()==False,'Description'].apply(lambda s: np.where("nan" in s.lower(), True, False)).value_counts()
data.loc[data.Description.isnull()==False,'Description'] = data.loc[data.Description.isnull()==False,'Description'].apply(lambda s: np.where("nan" in s.lower(), np.nan, s))
data = data[(data.CustomerID.isnull()==False) & (data.Description.isnull()==False)].copy()
d = data.groupby('Description').size().to_frame().reset_index().sort_values(by=0,ascending=False).head(25)
def count_numeric_chars(l):
    return sum(1 for c in l if c.isdigit())
data["StockCodeLength"] = data.StockCode.apply(lambda l: len(l))
data["nNumericStockCode"] = data.StockCode.apply(lambda l: count_numeric_chars(l))
data = data.loc[(data.nNumericStockCode == 5) & (data.StockCodeLength==5)].copy()
data["InvoiceDate"] = pd.to_datetime(data.InvoiceDate, cache=True)
data["Year"] = data.InvoiceDate.dt.year
data["Quarter"] = data.InvoiceDate.dt.quarter
data["Month"] = data.InvoiceDate.dt.month
data["Week"] = data.InvoiceDate.dt.isocalendar().week
data["Weekday"] = data.InvoiceDate.dt.weekday
data["Day"] = data.InvoiceDate.dt.day
data["Dayofyear"] = data.InvoiceDate.dt.dayofyear
data["Date"] = pd.to_datetime(data[['Year', 'Month', 'Day']])
grouped_features = ["Date", "Year", "Quarter","Month", "Week", "Weekday", "Dayofyear", "Day","StockCode"]
daily_data = pd.DataFrame(data.groupby(grouped_features).Quantity.sum(),columns=["Quantity"])
daily_data["Revenue"] = data.groupby(grouped_features).Revenue.sum()
daily_data = daily_data.reset_index()
daily_data.head(5)
daily_data.Quantity = remove_outliers(daily_data.Quantity)
daily_data.Revenue = remove_outliers(daily_data.Revenue)
daily_data.dropna(inplace=True)


#KNRegression Model
st.write(":red[KNeighborsRegressor Model:]")
X = daily_data.drop(['Date','Revenue','Quantity'],axis=1)
y = daily_data.Quantity
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2, shuffle=False)
param_grid = {'n_neighbors':list(range(3,100,2))}
m = KNeighborsRegressor()
search = GridSearchCV(m, param_grid, cv=5)
search.fit(X,y)
KNR = KNeighborsRegressor(n_neighbors=47)
KNR.fit(X_train,y_train)
KNRforecast = KNR.predict(X_valid)
X_valid['forecast'] = KNRforecast
spliter = round(len(daily_data)*0.8)
X_valid['Date'] = daily_data['Date'].iloc[spliter:]
preds= X_valid[['Date','forecast']].groupby('Date').mean()
true = daily_data[['Date','Quantity']].iloc[spliter:,:].groupby('Date').mean()
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=true.index, y=true['Quantity'], mode='lines', marker=dict(color='#783242'), name='Original Data'))
fig.add_trace(go.Scatter(x=true.index, y=preds['forecast'], mode='lines', name='Prediction'))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Quantity',
    autosize=False,
    width=950,
    height=600
)
st.plotly_chart(fig)
st.write("Akurasi prediksi: ",100-mean_squared_error(y_valid,KNRforecast))
#RFRegression Model
st.write(":red[RandomForestRegressor Model:]")
X = daily_data.drop(['Date','Revenue','Quantity'],axis=1)
y = daily_data.Quantity
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2, shuffle=False)
RFR = RandomForestRegressor(max_depth=20, n_estimators=100, random_state=0)
RFR.fit(X_train,y_train)
RFRforecast = RFR.predict(X_valid)
X_valid['forecast'] = RFRforecast
spliter = round(len(daily_data)*0.8)
X_valid['Date'] = daily_data['Date'].iloc[spliter:]
preds= X_valid[['Date','forecast']].groupby('Date').mean()
true = daily_data[['Date','Quantity']].iloc[spliter:,:].groupby('Date').mean()
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=true.index, y=true['Quantity'], mode='lines', marker=dict(color='#783242'), name='Original Data'))
fig.add_trace(go.Scatter(x=true.index, y=preds['forecast'], mode='lines', name='Prediction'))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Quantity',
    autosize=False,
    width=950,
    height=600
)
st.plotly_chart(fig)
st.write("Akurasi prediksi: ",100-mean_squared_error(y_valid,RFRforecast))
#CBRegressor Model
X = daily_data.drop(['Date','Revenue','Quantity'],axis=1)
y = daily_data.Quantity
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2, shuffle=False)
cat_features_idx = np.where(X.dtypes != int)[0]
train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
val_pool = Pool(X_valid, y_valid, cat_features=cat_features_idx)
CBR = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=0,
    logging_level='Silent',
    iterations=1000,
    max_depth=4,
    l2_leaf_reg=3,
    od_type='Iter',
    od_wait=40,
    train_dir="baseline",
    has_time=True
)
CBR.fit(train_pool, eval_set=val_pool, plot=True)
CBRforecast = CBR.predict(X_valid)
mse = mean_squared_error(y_valid, CBRforecast)
X_valid['forecast'] = CBRforecast
spliter = round(len(daily_data)*0.8)
X_valid['Date'] = daily_data['Date'].iloc[spliter:]
preds= X_valid[['Date','forecast']].groupby('Date').mean()
true = daily_data[['Date','Quantity']].iloc[spliter:,:].groupby('Date').mean()
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=true.index, y=true['Quantity'], mode='lines', marker=dict(color='#783242'), name='Original Data'))
fig.add_trace(go.Scatter(x=true.index, y=preds['forecast'], mode='lines', name='Prediction'))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Quantity',
    autosize=False,
    width=950,
    height=600
)
st.plotly_chart(fig)
st.write("Akurasi prediksi: ",100-mean_squared_error(y_valid,CBRforecast))


st.subheader("Kesimpulan:")
st.write('''
  Dari tiga model yang telah diterapkan, model CatBoostRegression memiliki akurasi tertinggi untuk kasus E-Commerce, yaitu sebesar 66,9995% 

  Akurasi 66,9995% termasuk rendah, sehingga perlu ditingkatkan lagi pengolahan datanya, atau dapat menerapkan model yang lain
''')



