import os
import traceback
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# APP & SERVER INITIALIZATION (Required for Vercel WSGI)
# ============================================================

dash_app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

server = dash_app.server

# CRITICAL FOR VERCEL: Vercel serverless function launcher looks for 'app'
app = server

dash_app.title = "Online Retail Intelligence Dashboard"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SALES_FILE = os.path.join(DATA_DIR, "online_retail_processed.csv")
PRODUCT_FILE = os.path.join(DATA_DIR, "product_segment.csv")
RFM_FILE = os.path.join(DATA_DIR, "rfm_segmented_customers.csv")


# ============================================================
# GLOBAL DATA
# ============================================================

analytics_df = pd.DataFrame()
DATA_ERROR = None


# ============================================================
# LOGGING
# ============================================================

print("=" * 80)
print("ONLINE RETAIL DASHBOARD")
print("=" * 80)
print("BASE DIR:", BASE_DIR)
print("DATA DIR:", DATA_DIR)
print("Sales CSV exists:", os.path.exists(SALES_FILE))
print("Product CSV exists:", os.path.exists(PRODUCT_FILE))
print("RFM CSV exists:", os.path.exists(RFM_FILE))
print("=" * 80)


# ============================================================
# LOAD & PREPARE DATA
# ============================================================

def load_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def find_column(df, possible_names):
    normalized = {str(col).strip().lower().replace(" ", "").replace("_", ""): col for col in df.columns}
    for name in possible_names:
        key = str(name).strip().lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


try:
    sales_raw = load_csv(SALES_FILE, "online_retail_processed.csv")
    product_raw = load_csv(PRODUCT_FILE, "product_segment.csv")
    rfm_raw = load_csv(RFM_FILE, "rfm_segmented_customers.csv")

    invoice_col = find_column(sales_raw, ["InvoiceNo", "Invoice"])
    stock_col = find_column(sales_raw, ["StockCode", "Stock"])
    description_col = find_column(sales_raw, ["Description"])
    quantity_col = find_column(sales_raw, ["Quantity", "Qty"])
    date_col = find_column(sales_raw, ["InvoiceDate", "Date"])
    price_col = find_column(sales_raw, ["UnitPrice", "Price"])
    customer_col = find_column(sales_raw, ["CustomerID", "CustomerId", "Customer"])
    country_col = find_column(sales_raw, ["Country"])
    total_col = find_column(sales_raw, ["total_amount", "TotalAmount", "Revenue", "Sales"])

    required = {
        "InvoiceNo": invoice_col,
        "StockCode": stock_col,
        "Quantity": quantity_col,
        "InvoiceDate": date_col,
        "UnitPrice": price_col,
        "Country": country_col
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("Missing required sales columns: " + ", ".join(missing))

    sales_df = sales_raw.copy()

    sales_df["InvoiceNo"] = sales_df[invoice_col].astype(str).str.strip()
    sales_df["StockCode"] = sales_df[stock_col].astype(str).str.strip()

    if description_col:
        sales_df["Description"] = sales_df[description_col].fillna("Unknown Product").astype(str).str.strip()
    else:
        sales_df["Description"] = sales_df["StockCode"]

    sales_df["Quantity"] = pd.to_numeric(sales_df[quantity_col], errors="coerce")
    sales_df["UnitPrice"] = pd.to_numeric(sales_df[price_col], errors="coerce")
    sales_df["InvoiceDate"] = pd.to_datetime(sales_df[date_col], errors="coerce")
    sales_df["Country"] = sales_df[country_col].fillna("Unknown").astype(str).str.strip()

    if customer_col:
        customer_numeric = pd.to_numeric(sales_df[customer_col], errors="coerce")
        sales_df["CustomerID"] = customer_numeric.apply(
            lambda x: str(int(x)) if pd.notna(x) else "Unknown"
        )
    else:
        sales_df["CustomerID"] = "Unknown"

    if total_col:
        sales_df["total_amount"] = pd.to_numeric(sales_df[total_col], errors="coerce")
    else:
        sales_df["total_amount"] = sales_df["Quantity"] * sales_df["UnitPrice"]

    sales_df = sales_df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice", "total_amount"])
    sales_df = sales_df[~sales_df["InvoiceNo"].str.upper().str.startswith("C")]
    sales_df = sales_df[
        (sales_df["Quantity"] > 0)
        & (sales_df["UnitPrice"] > 0)
        & (sales_df["total_amount"] > 0)
    ]

    sales_df["Date"] = sales_df["InvoiceDate"].dt.normalize()
    sales_df["Year"] = sales_df["InvoiceDate"].dt.year
    sales_df["Month"] = sales_df["InvoiceDate"].dt.to_period("M").astype(str)
    sales_df["DayName"] = sales_df["InvoiceDate"].dt.day_name()

    product_df = product_raw.copy()
    product_stock_col = find_column(product_df, ["StockCode", "Stock"])
    category_col = find_column(product_df, ["Category", "ProductCategory"])
    price_segment_col = find_column(product_df, ["PriceSegment", "Price Segment", "Price_Segment"])

    if product_stock_col:
        product_lookup = product_df.copy()
        product_lookup["_StockCode"] = product_lookup[product_stock_col].astype(str).str.strip()

        keep_columns = ["_StockCode"]
        if category_col:
            keep_columns.append(category_col)
        if price_segment_col:
            keep_columns.append(price_segment_col)

        product_lookup = product_lookup[keep_columns].drop_duplicates(subset=["_StockCode"])

        rename_map = {"_StockCode": "StockCode"}
        if category_col:
            rename_map[category_col] = "Category"
        if price_segment_col:
            rename_map[price_segment_col] = "PriceSegment"

        product_lookup = product_lookup.rename(columns=rename_map)
        sales_df = sales_df.merge(product_lookup, on="StockCode", how="left")

    if "Category" not in sales_df.columns:
        sales_df["Category"] = "Other"

    sales_df["Category"] = sales_df["Category"].fillna("Other").astype(str).str.strip()

    if "PriceSegment" not in sales_df.columns:
        sales_df["PriceSegment"] = pd.cut(
            sales_df["UnitPrice"],
            bins=[-np.inf, 2, 5, 10, 25, np.inf],
            labels=["Budget", "Low", "Medium", "Premium", "Luxury"]
        )

    sales_df["PriceSegment"] = sales_df["PriceSegment"].fillna("Unknown").astype(str)

    rfm_df = rfm_raw.copy()
    rfm_customer_col = find_column(rfm_df, ["CustomerID", "CustomerId", "Customer"])
    rfm_segment_col = find_column(rfm_df, ["Segment", "CustomerSegment", "Customer_Segment"])
    recency_col = find_column(rfm_df, ["Recency"])
    frequency_col = find_column(rfm_df, ["Frequency"])
    monetary_col = find_column(rfm_df, ["Monetary", "MonetaryValue", "Monetary_Value"])

    if rfm_customer_col:
        rfm_numeric = pd.to_numeric(rfm_df[rfm_customer_col], errors="coerce")
        rfm_df["CustomerID"] = rfm_numeric.apply(
            lambda x: str(int(x)) if pd.notna(x) else "Unknown"
        )
    else:
        rfm_df["CustomerID"] = "Unknown"

    if rfm_segment_col:
        rfm_df["Segment"] = rfm_df[rfm_segment_col].fillna("Unknown").astype(str).str.strip()
    else:
        rfm_df["Segment"] = "Unknown"

    rfm_df["Recency"] = pd.to_numeric(rfm_df[recency_col], errors="coerce") if recency_col else np.nan
    rfm_df["Frequency"] = pd.to_numeric(rfm_df[frequency_col], errors="coerce") if frequency_col else np.nan
    rfm_df["Monetary"] = pd.to_numeric(rfm_df[monetary_col], errors="coerce") if monetary_col else np.nan

    rfm_lookup = rfm_df[["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]].drop_duplicates(
        subset=["CustomerID"]
    )

    analytics_df = sales_df.merge(rfm_lookup, on="CustomerID", how="left")
    analytics_df["Segment"] = analytics_df["Segment"].fillna("Unknown Customer").astype(str)
    analytics_df["Recency"] = pd.to_numeric(analytics_df["Recency"], errors="coerce")
    analytics_df["Frequency"] = pd.to_numeric(analytics_df["Frequency"], errors="coerce")
    analytics_df["Monetary"] = pd.to_numeric(analytics_df["Monetary"], errors="coerce")
    analytics_df = analytics_df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    if analytics_df.empty:
        raise ValueError("Analytics dataframe is empty.")

    print("DATA PREPARATION SUCCESS - Rows loaded:", len(analytics_df))

except Exception as error:
    DATA_ERROR = str(error)
    print("DATA PREPARATION FAILED:", DATA_ERROR)
    traceback.print_exc()


# ============================================================
# DATE RANGE / FILTER OPTIONS
# ============================================================

if not analytics_df.empty:
    MIN_DATE = analytics_df["InvoiceDate"].min().date()
    MAX_DATE = analytics_df["InvoiceDate"].max().date()
    COUNTRY_OPTIONS = sorted(analytics_df["Country"].dropna().unique().tolist())
    CATEGORY_OPTIONS = sorted(analytics_df["Category"].dropna().unique().tolist())
    PRICE_OPTIONS = sorted(analytics_df["PriceSegment"].dropna().unique().tolist())
    SEGMENT_OPTIONS = sorted(analytics_df["Segment"].dropna().unique().tolist())
else:
    MIN_DATE = None
    MAX_DATE = None
    COUNTRY_OPTIONS = []
    CATEGORY_OPTIONS = []
    PRICE_OPTIONS = []
    SEGMENT_OPTIONS = []


# ============================================================
# HELPERS & FILTER LOGIC
# ============================================================

def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16)
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=50, r=30, t=60, b=50))
    return fig


def style_graph(fig):
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=50, r=30, t=60, b=50), hovermode="closest")
    return fig


def create_kpi(title, value):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value")
        ]),
        className="kpi-card"
    )


def filter_data(df, countries, categories, prices, segments, start_date, end_date):
    result = df.copy()

    if countries:
        result = result[result["Country"].isin(countries)]
    if categories:
        result = result[result["Category"].isin(categories)]
    if prices:
        result = result[result["PriceSegment"].isin(prices)]
    if segments:
        result = result[result["Segment"].isin(segments)]

    if start_date:
        start = pd.to_datetime(start_date)
        result = result[result["InvoiceDate"] >= start]

    if end_date:
        end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        result = result[result["InvoiceDate"] < end]

    return result


def get_filtered_df(countries, categories, prices, segments, start_date, end_date):
    return filter_data(analytics_df, countries, categories, prices, segments, start_date, end_date)


# ============================================================
# LAYOUT
# ============================================================

def filter_row():
    return dbc.Card(
        dbc.CardBody([
            html.H5("Dashboard Filters"),
            dbc.Row([
                dbc.Col([
                    html.Label("Country"),
                    dcc.Dropdown(id="country-filter", options=[{"label": x, "value": x} for x in COUNTRY_OPTIONS], multi=True, placeholder="All Countries")
                ], md=2),
                dbc.Col([
                    html.Label("Product Category"),
                    dcc.Dropdown(id="category-filter", options=[{"label": x, "value": x} for x in CATEGORY_OPTIONS], multi=True, placeholder="All Categories")
                ], md=2),
                dbc.Col([
                    html.Label("Price Segment"),
                    dcc.Dropdown(id="price-filter", options=[{"label": x, "value": x} for x in PRICE_OPTIONS], multi=True, placeholder="All Price Segments")
                ], md=2),
                dbc.Col([
                    html.Label("Customer Segment"),
                    dcc.Dropdown(id="customer-segment-filter", options=[{"label": x, "value": x} for x in SEGMENT_OPTIONS], multi=True, placeholder="All Customer Segments")
                ], md=2),
                dbc.Col([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(id="date-filter", start_date=MIN_DATE, end_date=MAX_DATE, min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE, display_format="YYYY-MM-DD")
                ], md=3),
                dbc.Col([
                    html.Label("Action"),
                    dbc.Button("Reset", id="reset-button", color="primary", className="w-100")
                ], md=1)
            ])
        ]),
        className="mb-3"
    )


dash_app.layout = dbc.Container(
    [
        dcc.Interval(id="startup-trigger", interval=500, n_intervals=0, max_intervals=1),
        html.Div([
            html.H2("Online Retail Intelligence Dashboard", className="dashboard-title"),
            html.P("Customer, Product and Sales Analytics", className="dashboard-subtitle")
        ], className="dashboard-header"),
        dbc.Alert(
            "Data loaded successfully." if not DATA_ERROR else f"Data loading failed: {DATA_ERROR}",
            color="success" if not DATA_ERROR else "danger",
            className="mb-3"
        ),
        filter_row(),
        dbc.Row([
            dbc.Col(html.Div(id="kpi-revenue"), md=3),
            dbc.Col(html.Div(id="kpi-orders"), md=3),
            dbc.Col(html.Div(id="kpi-customers"), md=3),
            dbc.Col(html.Div(id="kpi-products"), md=3),
        ], className="mb-3"),
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-exec",
            children=[
                dbc.Tab(
                    label="Executive Overview", tab_id="tab-exec",
                    children=[
                        dbc.Row([dbc.Col(dcc.Graph(id="monthly-sales"), md=8), dbc.Col(dcc.Graph(id="country-sales"), md=4)]),
                        dbc.Row([dbc.Col(dcc.Graph(id="category-sales"), md=6), dbc.Col(dcc.Graph(id="segment-distribution"), md=6)])
                    ]
                ),
                dbc.Tab(
                    label="Customer Intelligence", tab_id="tab-customer",
                    children=[
                        dbc.Row([dbc.Col(dcc.Graph(id="rfm-scatter"), md=6), dbc.Col(dcc.Graph(id="customer-revenue"), md=6)]),
                        dbc.Row([dbc.Col(dcc.Graph(id="recency-frequency"), md=6), dbc.Col(dcc.Graph(id="segment-revenue"), md=6)])
                    ]
                ),
                dbc.Tab(
                    label="Product Intelligence", tab_id="tab-product",
                    children=[
                        dbc.Row([dbc.Col(dcc.Graph(id="top-products"), md=6), dbc.Col(dcc.Graph(id="price-segment-sales"), md=6)]),
                        dbc.Row([dbc.Col(dcc.Graph(id="quantity-category"), md=6), dbc.Col(dcc.Graph(id="category-price"), md=6)])
                    ]
                ),
                dbc.Tab(
                    label="Sales Analytics", tab_id="tab-sales",
                    children=[
                        dbc.Row([dbc.Col(dcc.Graph(id="daily-sales"), md=8), dbc.Col(dcc.Graph(id="weekday-sales"), md=4)]),
                        dbc.Row([dbc.Col(dcc.Graph(id="quantity-revenue"), md=6), dbc.Col(dcc.Graph(id="order-value"), md=6)])
                    ]
                )
            ]
        ),
        html.Hr(),
        html.Div("Online Retail Intelligence Dashboard", className="footer")
    ],
    fluid=True
)


# ============================================================
# CALLBACKS
# ============================================================

FILTER_INPUTS = [
    Input("country-filter", "value"),
    Input("category-filter", "value"),
    Input("price-filter", "value"),
    Input("customer-segment-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
]


def error_kpis():
    card = create_kpi("Error", "ERROR")
    return card, card, card, card


def error_figs(n):
    fig = empty_figure("Callback error - check logs")
    return tuple(fig for _ in range(n))


@dash_app.callback(
    [
        Output("kpi-revenue", "children"),
        Output("kpi-orders", "children"),
        Output("kpi-customers", "children"),
        Output("kpi-products", "children"),
    ],
    [Input("startup-trigger", "n_intervals")] + FILTER_INPUTS
)
def update_kpis(n_intervals, countries, categories, prices, segments, start_date, end_date):
    if DATA_ERROR or analytics_df.empty:
        return error_kpis()
    try:
        df = get_filtered_df(countries, categories, prices, segments, start_date, end_date)
        if df.empty:
            return create_kpi("Total Revenue", "£0"), create_kpi("Total Orders", "0"), create_kpi("Total Customers", "0"), create_kpi("Total Products", "0")
        
        revenue = df["total_amount"].sum()
        orders = df["InvoiceNo"].nunique()
        customers = df[df["CustomerID"] != "Unknown"]["CustomerID"].nunique()
        products = df["StockCode"].nunique()

        return (
            create_kpi("Total Revenue", f"£{revenue:,.0f}"),
            create_kpi("Total Orders", f"{orders:,}"),
            create_kpi("Total Customers", f"{customers:,}"),
            create_kpi("Total Products", f"{products:,}"),
        )
    except Exception as error:
        return error_kpis()


@dash_app.callback(
    [
        Output("monthly-sales", "figure"),
        Output("country-sales", "figure"),
        Output("category-sales", "figure"),
        Output("segment-distribution", "figure"),
    ],
    [Input("main-tabs", "active_tab")] + FILTER_INPUTS
)
def update_executive_tab(active_tab, countries, categories, prices, segments, start_date, end_date):
    if DATA_ERROR or analytics_df.empty:
        return error_figs(4)
    try:
        df = get_filtered_df(countries, categories, prices, segments, start_date, end_date)
        if df.empty:
            fig = empty_figure("No data available for selected filters")
            return fig, fig, fig, fig

        monthly = df.groupby("Month", as_index=False)["total_amount"].sum().sort_values("Month")
        fig_monthly = style_graph(px.line(monthly, x="Month", y="total_amount", markers=True, title="Monthly Revenue"))

        country = df.groupby("Country", as_index=False)["total_amount"].sum().sort_values("total_amount", ascending=False).head(10)
        fig_country = style_graph(px.bar(country, x="total_amount", y="Country", orientation="h", title="Top Countries by Revenue"))

        category = df.groupby("Category", as_index=False)["total_amount"].sum().sort_values("total_amount", ascending=False)
        fig_category = style_graph(px.bar(category, x="Category", y="total_amount", title="Revenue by Category"))

        segment = df[df["CustomerID"] != "Unknown"].groupby("Segment", as_index=False)["CustomerID"].nunique()
        fig_segment = style_graph(px.pie(segment, names="Segment", values="CustomerID", title="Customer Segment Distribution")) if not segment.empty else empty_figure("No customer segment data")

        return fig_monthly, fig_country, fig_category, fig_segment
    except Exception as error:
        return error_figs(4)


@dash_app.callback(
    [
        Output("rfm-scatter", "figure"),
        Output("customer-revenue", "figure"),
        Output("recency-frequency", "figure"),
        Output("segment-revenue", "figure"),
    ],
    [Input("main-tabs", "active_tab")] + FILTER_INPUTS
)
def update_customer_tab(active_tab, countries, categories, prices, segments, start_date, end_date):
    if DATA_ERROR or analytics_df.empty:
        return error_figs(4)
    try:
        df = get_filtered_df(countries, categories, prices, segments, start_date, end_date)
        if df.empty:
            fig = empty_figure("No data available for selected filters")
            return fig, fig, fig, fig

        known = df[df["CustomerID"] != "Unknown"]

        rfm_plot = known.groupby(["CustomerID", "Segment"], as_index=False).agg(
            Recency=("Recency", "first"), Frequency=("Frequency", "first"), Monetary=("Monetary", "first")
        ).dropna(subset=["Frequency", "Monetary"])
        fig_rfm = style_graph(px.scatter(rfm_plot, x="Frequency", y="Monetary", color="Segment", hover_data=["CustomerID", "Recency"], title="RFM Customer Analysis")) if not rfm_plot.empty else empty_figure("No RFM data")

        customer_revenue = known.groupby("CustomerID", as_index=False)["total_amount"].sum().sort_values("total_amount", ascending=False).head(15)
        fig_customer = style_graph(px.bar(customer_revenue, x="CustomerID", y="total_amount", title="Top Customers by Revenue"))

        recency_frequency = known.groupby("CustomerID", as_index=False).agg(
            Recency=("Recency", "first"), Frequency=("Frequency", "first"), Segment=("Segment", "first")
        ).dropna(subset=["Recency", "Frequency"])
        fig_recency = style_graph(px.scatter(recency_frequency, x="Recency", y="Frequency", color="Segment", hover_data=["CustomerID"], title="Recency vs Frequency")) if not recency_frequency.empty else empty_figure("No recency/frequency data")

        segment_revenue = known.groupby("Segment", as_index=False)["total_amount"].sum().sort_values("total_amount", ascending=False)
        fig_segment_revenue = style_graph(px.bar(segment_revenue, x="Segment", y="total_amount", title="Revenue by Customer Segment"))

        return fig_rfm, fig_customer, fig_recency, fig_segment_revenue
    except Exception as error:
        return error_figs(4)


@dash_app.callback(
    [
        Output("top-products", "figure"),
        Output("price-segment-sales", "figure"),
        Output("quantity-category", "figure"),
        Output("category-price", "figure"),
    ],
    [Input("main-tabs", "active_tab")] + FILTER_INPUTS
)
def update_product_tab(active_tab, countries, categories, prices, segments, start_date, end_date):
    if DATA_ERROR or analytics_df.empty:
        return error_figs(4)
    try:
        df = get_filtered_df(countries, categories, prices, segments, start_date, end_date)
        if df.empty:
            fig = empty_figure("No data available for selected filters")
            return fig, fig, fig, fig

        top_products = df.groupby("Description", as_index=False).agg(
            Revenue=("total_amount", "sum"), Quantity=("Quantity", "sum")
        ).sort_values("Revenue", ascending=False).head(15)
        fig_products = style_graph(px.bar(top_products, x="Revenue", y="Description", orientation="h", title="Top Products by Revenue"))

        price_sales = df.groupby("PriceSegment", as_index=False)["total_amount"].sum()
        fig_price = style_graph(px.bar(price_sales, x="PriceSegment", y="total_amount", title="Revenue by Price Segment"))

        quantity_category = df.groupby("Category", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
        fig_quantity = style_graph(px.bar(quantity_category, x="Category", y="Quantity", title="Quantity Sold by Category"))

        category_price = df.groupby(["Category", "PriceSegment"], as_index=False)["total_amount"].sum()
        fig_category_price = style_graph(px.bar(category_price, x="Category", y="total_amount", color="PriceSegment", barmode="group", title="Category vs Price Segment"))

        return fig_products, fig_price, fig_quantity, fig_category_price
    except Exception as error:
        return error_figs(4)


@dash_app.callback(
    [
        Output("daily-sales", "figure"),
        Output("weekday-sales", "figure"),
        Output("quantity-revenue", "figure"),
        Output("order-value", "figure"),
    ],
    [Input("main-tabs", "active_tab")] + FILTER_INPUTS
)
def update_sales_tab(active_tab, countries, categories, prices, segments, start_date, end_date):
    if DATA_ERROR or analytics_df.empty:
        return error_figs(4)
    try:
        df = get_filtered_df(countries, categories, prices, segments, start_date, end_date)
        if df.empty:
            fig = empty_figure("No data available for selected filters")
            return fig, fig, fig, fig

        daily = df.groupby("Date", as_index=False)["total_amount"].sum()
        fig_daily = style_graph(px.line(daily, x="Date", y="total_amount", title="Daily Revenue"))

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday = df.groupby("DayName", as_index=False)["total_amount"].sum()
        weekday["DayName"] = pd.Categorical(weekday["DayName"], categories=weekday_order, ordered=True)
        fig_weekday = style_graph(px.bar(weekday.sort_values("DayName"), x="DayName", y="total_amount", title="Revenue by Weekday"))

        quantity_revenue = df.groupby(["StockCode", "Description"], as_index=False).agg(
            Quantity=("Quantity", "sum"), Revenue=("total_amount", "sum")
        ).sort_values("Revenue", ascending=False).head(200)
        fig_quantity_revenue = style_graph(px.scatter(quantity_revenue, x="Quantity", y="Revenue", size="Revenue", hover_data=["Description"], title="Quantity vs Revenue"))

        order_value = df.groupby("InvoiceNo", as_index=False)["total_amount"].sum()
        fig_order_value = style_graph(px.histogram(order_value, x="total_amount", nbins=30, title="Order Value Distribution"))

        return fig_daily, fig_weekday, fig_quantity_revenue, fig_order_value
    except Exception as error:
        return error_figs(4)


@dash_app.callback(
    [
        Output("country-filter", "value"),
        Output("category-filter", "value"),
        Output("price-filter", "value"),
        Output("customer-segment-filter", "value"),
        Output("date-filter", "start_date"),
        Output("date-filter", "end_date"),
    ],
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return None, None, None, None, MIN_DATE, MAX_DATE


dash_app.clientside_callback(
    """
    function(active_tab) {
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
        }, 50);
        return window.dash_clientside.no_update;
    }
    """,
    Output("main-tabs", "id"),
    Input("main-tabs", "active_tab"),
    prevent_initial_call=True
)


@server.route("/health")
def health():
    if DATA_ERROR:
        return {"status": "error", "error": DATA_ERROR}
    return {
        "status": "ok",
        "sales_rows": int(len(analytics_df)),
        "orders": int(analytics_df["InvoiceNo"].nunique()) if not analytics_df.empty else 0,
        "revenue": float(analytics_df["total_amount"].sum()) if not analytics_df.empty else 0.0,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    dash_app.run(host="0.0.0.0", port=port, debug=False)
