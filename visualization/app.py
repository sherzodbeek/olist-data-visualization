"""Olist business dashboard: Sales and Marketplace Operations, Logistics and Fulfillment,
and Customer Experience.

Run data_preparation.ipynb first, then start the dashboard from this folder:
    ../.venv/bin/streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
PROCESSED_FILES = ["orders.parquet", "order_items.parquet", "states.parquet"]

REGIONS = ["North", "Northeast", "Central-West", "Southeast", "South"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STAGES = {
    "approval_days": "Payment approval",
    "handling_days": "Seller handling",
    "transit_days": "Carrier transit",
}

st.set_page_config(page_title="Olist Business Dashboard", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Colors: validated light and dark steps of one palette. Categorical slots 1-3,
# a one-hue sequential ramp, a red <-> gray <-> blue diverging ramp for review
# scores, and the reserved "late" status color.
# ---------------------------------------------------------------------------
LIGHT = {
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#ffffff",
    "land": "#f0efec",
    "neutral": "#b5b4ac",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "late": "#d03b3b",
    "scores": {1: "#c93a3a", 2: "#ec9a98", 3: "#b5b4ac", 4: "#86b6ef", 5: "#2a78d6"},
    "sequential": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
}
DARK = {
    "text": "#f0efec",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "surface": "#0e1117",
    "land": "#262624",
    "neutral": "#5d5c57",
    "series": ["#3987e5", "#d95926", "#199e70"],
    "late": "#d03b3b",
    "scores": {1: "#f09a98", 2: "#a83b3b", 3: "#5d5c57", 4: "#256abf", 5: "#86b6ef"},
    "sequential": ["#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
}
COLORS = DARK if st.context.theme.type == "dark" else LIGHT
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def brl(value: float) -> str:
    if pd.isna(value):
        return "–"
    if abs(value) >= 1_000_000:
        return f"R$ {value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"R$ {value / 1_000:.1f}K"
    return f"R$ {value:,.2f}"


def percent(value: float) -> str:
    return "–" if pd.isna(value) else f"{value:.1%}"


def days(value: float) -> str:
    return "–" if pd.isna(value) else f"{value:.1f} days"


def count(value: float) -> str:
    return "–" if pd.isna(value) else f"{value:,.0f}"


def stars(value: float) -> str:
    return "–" if pd.isna(value) else f"{value:.2f} ★"


def ink_on(fill: str) -> str:
    """Label color that stays readable inside a bar of the given fill color."""
    red, green, blue = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#0b0b0b" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.6 else "#ffffff"


def with_alpha(hex_color: str, alpha: float) -> str:
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def line_mode(points: int) -> str:
    # A line needs two points; show markers when a short period leaves too few.
    return "lines" if points > 2 else "lines+markers"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_parquet(PROCESSED_DIR / "orders.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "order_items.parquet")
    states = pd.read_parquet(PROCESSED_DIR / "states.parquet")
    # Nullable booleans become 1.0 / 0.0 / NaN so means and group means skip missing values.
    for column in ["is_late", "is_interstate", "has_comment"]:
        orders[column] = orders[column].astype("Float64").astype(float)
    return orders, items, states


def summarize(orders: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for one slice of orders, used for KPI tiles and their deltas."""
    sales = orders[orders["is_sale"]]
    delivered = orders[orders["is_delivered"]]
    reviewed = orders[orders["review_score"].notna()]
    orders_per_customer = sales.groupby("customer_unique_id")["order_id"].count()
    return {
        "revenue": sales["order_value"].sum(),
        "orders": len(sales),
        "aov": sales["order_value"].mean(),
        "customers": sales["customer_unique_id"].nunique(),
        "product_revenue": sales["product_value"].sum(),
        "freight_revenue": sales["freight_value"].sum(),
        "items": sales["item_count"].sum(),
        "items_per_order": sales["item_count"].mean(),
        "delivered": len(delivered),
        "delivery_days": delivered["delivery_days"].mean(),
        "estimated_days": delivered["estimated_days"].mean(),
        "on_time_rate": 1 - delivered["is_late"].mean(),
        "late_rate": delivered["is_late"].mean(),
        "freight_share": sales["freight_share"].mean(),
        "interstate_rate": sales["is_interstate"].mean(),
        "score": reviewed["review_score"].mean(),
        "five_star_rate": (reviewed["review_score"] == 5).mean(),
        "low_score_rate": (reviewed["review_score"] <= 2).mean(),
        "comment_rate": reviewed["has_comment"].mean(),
        "response_days": reviewed["review_response_days"].mean(),
        "repeat_rate": (orders_per_customer > 1).mean(),
    }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int, horizontal: bool = False, margin_top: int = 8) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=12, t=margin_top, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=COLORS["text_secondary"]),
        hoverlabel=dict(font_family=FONT),
        legend=dict(orientation="h", x=0, y=1.0, xanchor="left", yanchor="bottom", title_text=""),
        barcornerradius=4,
        bargap=0.3,
    )
    value_axis = dict(gridcolor=COLORS["grid"], zeroline=False, tickfont_color=COLORS["muted"], automargin=True)
    category_axis = dict(showgrid=False, linecolor=COLORS["axis"], ticks="", tickfont_color=COLORS["text_secondary"],
                         automargin=True)
    fig.update_xaxes(**(value_axis if horizontal else category_axis))
    fig.update_yaxes(**(category_axis if horizontal else value_axis))
    return fig


def chart_card(title: str, caption: str, fig: go.Figure, table: pd.DataFrame,
               height: int = 340, horizontal: bool = False, margin_top: int = 8) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(caption)
        st.plotly_chart(style(fig, height, horizontal, margin_top), width="stretch", theme=None,
                        config={"displayModeBar": False})
        with st.expander("Table view"):
            st.dataframe(table, hide_index=True, width="stretch")


def kpi(column, label: str, key: str, current: dict, previous: dict | None, fmt,
        delta: str = "relative", lower_is_better: bool = False, help: str | None = None) -> None:
    """A stat tile; the delta compares with the previous period of the same length."""
    value = current[key]
    delta_text = None
    if previous is not None and pd.notna(value) and pd.notna(previous[key]):
        before = previous[key]
        if delta == "relative" and before != 0:
            delta_text = f"{(value - before) / before:+.1%}"
        elif delta == "points":
            delta_text = f"{(value - before) * 100:+.1f} pp"
        elif delta == "absolute":
            delta_text = f"{value - before:+.2f}"
    column.metric(label, fmt(value), delta_text, delta_color="inverse" if lower_is_better else "normal",
                  help=help, border=True)


def monthly_line(data: pd.DataFrame, y: str, hover_format: str, color: str, fill: bool = False) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=data["purchase_month"], y=data[y], mode=line_mode(len(data)),
        line=dict(color=color, width=2), marker=dict(size=8),
        fill="tozeroy" if fill else None, fillcolor=with_alpha(color, 0.12),
        hovertemplate=f"%{{x|%b %Y}}<br>{hover_format}<extra></extra>",
    ))
    fig.update_layout(hovermode="x", showlegend=False)
    fig.update_xaxes(tickformat="%b\n%Y", hoverformat="%b %Y")
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def filter_bar(orders: pd.DataFrame, states: pd.DataFrame):
    months = sorted(orders["purchase_month"].drop_duplicates())
    state_names = dict(zip(states["state"], states["state_name"]))

    period_col, region_col, state_col, category_col = st.columns([2.2, 1.3, 1.5, 2])
    start, end = period_col.select_slider(
        "Purchase period", options=months, value=(months[0], months[-1]),
        format_func=lambda month: month.strftime("%b %Y"),
    )
    regions = region_col.multiselect("Customer region", REGIONS, placeholder="All regions")
    state_options = states.loc[states["region"].isin(regions or REGIONS), "state"].tolist()
    selected_states = state_col.multiselect(
        "Customer state", state_options, placeholder="All states",
        format_func=lambda code: f"{code} – {state_names[code]}",
    )
    categories = category_col.multiselect(
        "Product category", sorted(orders["main_category"].unique()), placeholder="All categories",
    )

    other_filters = pd.Series(True, index=orders.index)
    if regions:
        other_filters &= orders["customer_region"].isin(regions)
    if selected_states:
        other_filters &= orders["customer_state"].isin(selected_states)
    if categories:
        other_filters &= orders["main_category"].isin(categories)

    current = orders[other_filters & orders["purchase_month"].between(start, end)]

    # Previous period of the same length, for KPI deltas (None when it falls before the data starts).
    length = months.index(end) - months.index(start) + 1
    first = months.index(start) - length
    previous = None
    if first >= 0:
        previous = orders[other_filters & orders["purchase_month"].between(months[first], months[first + length - 1])]

    period_label = f"{start:%b %Y} – {end:%b %Y}"
    return current, previous, categories, period_label


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def overview_tab(orders: pd.DataFrame, now: dict, before: dict | None) -> None:
    tiles = st.columns(6)
    kpi(tiles[0], "Revenue", "revenue", now, before, brl,
        help="Product price + freight paid by customers, excluding canceled and unavailable orders")
    kpi(tiles[1], "Orders", "orders", now, before, count)
    kpi(tiles[2], "Avg order value", "aov", now, before, brl)
    kpi(tiles[3], "Customers", "customers", now, before, count, help="Unique customers who placed an order")
    kpi(tiles[4], "On-time delivery", "on_time_rate", now, before, percent, delta="points",
        help="Share of delivered orders that arrived on or before the promised date")
    kpi(tiles[5], "Avg review score", "score", now, before, stars, delta="absolute")

    sales = orders[orders["is_sale"]]
    monthly = (
        sales.groupby("purchase_month")
        .agg(revenue=("order_value", "sum"), orders=("order_id", "count"))
        .reset_index()
    )
    left, right = st.columns(2)
    with left:
        chart_card(
            "Monthly revenue", "Product + freight, R$",
            monthly_line(monthly, "revenue", "R$ %{y:,.0f}", COLORS["series"][0], fill=True),
            monthly.assign(purchase_month=monthly["purchase_month"].dt.strftime("%b %Y")).round(2),
        )
    with right:
        chart_card(
            "Monthly orders", "Orders placed, excluding canceled and unavailable",
            monthly_line(monthly, "orders", "%{y:,} orders", COLORS["series"][0]),
            monthly.assign(purchase_month=monthly["purchase_month"].dt.strftime("%b %Y")).round(2),
        )

    reviewed = orders[orders["is_delivered"] & orders["review_score"].notna()]
    by_delay = (
        reviewed.groupby("delay_bucket", observed=True)
        .agg(avg_score=("review_score", "mean"), reviews=("order_id", "count"))
        .reset_index()
    )
    by_delay["delay_bucket"] = by_delay["delay_bucket"].astype(str)
    by_delay["is_late"] = by_delay["delay_bucket"].str.endswith("late")

    on_time_score = reviewed.loc[reviewed["is_late"] == 0, "review_score"].mean()
    late_score = reviewed.loc[reviewed["is_late"] == 1, "review_score"].mean()

    fig = go.Figure()
    for is_late, name, color in [(False, "On time or early", COLORS["series"][0]), (True, "Late", COLORS["late"])]:
        part = by_delay[by_delay["is_late"] == is_late]
        fig.add_bar(
            x=part["delay_bucket"], y=part["avg_score"], name=name, marker_color=color,
            customdata=part["reviews"],
            hovertemplate="%{x}<br>%{y:.2f} ★ average · %{customdata:,} reviews<extra></extra>",
        )
    fig.update_xaxes(categoryorder="array", categoryarray=by_delay["delay_bucket"].tolist())
    fig.update_yaxes(range=[0, 5], dtick=1, title_text="Average review score")

    left, right = st.columns([3, 2])
    with left:
        chart_card(
            "Late deliveries drive bad reviews",
            f"Average score {stars(on_time_score)} when on time vs {stars(late_score)} when late. "
            "Delivery date compared with the date promised at checkout.",
            fig, by_delay[["delay_bucket", "avg_score", "reviews"]].round(2),
        )
    with right:
        region_table = (
            orders.assign(sale_value=orders["order_value"].where(orders["is_sale"], 0))
            .groupby("customer_region")
            .agg(
                orders=("is_sale", "sum"),
                revenue=("sale_value", "sum"),
                avg_delivery_days=("delivery_days", "mean"),
                late_rate=("is_late", "mean"),
                avg_score=("review_score", "mean"),
            )
            .reindex(REGIONS).dropna(how="all")
            .sort_values("revenue", ascending=False)
            .reset_index()
        )
        with st.container(border=True):
            st.markdown("**Performance by customer region**")
            st.caption("All three departments side by side. Click a column header to sort.")
            st.dataframe(
                region_table, hide_index=True, width="stretch",
                column_config={
                    "customer_region": "Region",
                    "orders": st.column_config.NumberColumn("Orders", format="localized"),
                    "revenue": st.column_config.NumberColumn("Revenue (R$)", format="localized"),
                    "avg_delivery_days": st.column_config.NumberColumn("Delivery (d)", format="%.1f",
                                                                       help="Average delivery days"),
                    "late_rate": st.column_config.NumberColumn("Late", format="percent"),
                    "avg_score": st.column_config.NumberColumn("Score", format="%.2f ★"),
                },
            )


def sales_tab(orders: pd.DataFrame, items: pd.DataFrame, states: pd.DataFrame, now: dict, before: dict | None) -> None:
    sales = orders[orders["is_sale"]]
    sale_items = items[items["is_sale"]]

    tiles = st.columns(6)
    kpi(tiles[0], "Product revenue", "product_revenue", now, before, brl)
    kpi(tiles[1], "Freight revenue", "freight_revenue", now, before, brl)
    kpi(tiles[2], "Items sold", "items", now, before, count)
    kpi(tiles[3], "Items per order", "items_per_order", now, before, lambda value: f"{value:.2f}", delta="absolute")
    tiles[4].metric("Active sellers", count(sale_items["seller_id"].nunique()), border=True,
                    help="Sellers with at least one item sold in the selection")
    tiles[5].metric("Categories sold", count(sale_items["category"].nunique()), border=True)

    # Top categories and revenue by state
    categories = (
        sale_items.groupby("category")
        .agg(revenue=("item_value", "sum"), items=("order_item_id", "count"), orders=("order_id", "nunique"),
             avg_price=("price", "mean"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    top_categories = categories.head(15).iloc[::-1]
    fig_categories = go.Figure(go.Bar(
        x=top_categories["revenue"], y=top_categories["category"], orientation="h",
        marker_color=COLORS["series"][0], customdata=top_categories[["items"]],
        hovertemplate="%{y}<br>R$ %{x:,.0f} · %{customdata[0]:,} items<extra></extra>",
    ))
    fig_categories.update_xaxes(tickprefix="R$ ")

    state_names = dict(zip(states["state"], states["state_name"]))
    by_state = (
        sales.groupby("customer_state")
        .agg(revenue=("order_value", "sum"), orders=("order_id", "count"), aov=("order_value", "mean"))
        .sort_values("revenue")
        .reset_index()
    )
    by_state["state_name"] = by_state["customer_state"].map(state_names)
    fig_states = go.Figure(go.Bar(
        x=by_state["revenue"], y=by_state["customer_state"], orientation="h",
        marker_color=COLORS["series"][0], customdata=by_state[["state_name", "orders", "aov"]],
        hovertemplate="%{customdata[0]}<br>R$ %{x:,.0f} · %{customdata[1]:,} orders"
                      "<br>Avg order R$ %{customdata[2]:,.2f}<extra></extra>",
    ))
    fig_states.update_xaxes(tickprefix="R$ ")

    left, right = st.columns(2)
    with left:
        chart_card("Top 15 categories by revenue", "Item price + freight, by product category",
                   fig_categories, categories.round(2), height=480, horizontal=True)
    with right:
        chart_card("Revenue by customer state", "Where buyers are located",
                   fig_states, by_state.sort_values("revenue", ascending=False).round(2),
                   height=max(300, 18 * len(by_state) + 40), horizontal=True)

    # Average order value trend and purchase timing
    monthly_aov = sales.groupby("purchase_month")["order_value"].mean().rename("aov").reset_index()
    fig_aov = monthly_line(monthly_aov, "aov", "R$ %{y:,.2f}", COLORS["series"][0])
    fig_aov.update_yaxes(tickprefix="R$ ", rangemode="tozero")

    heat = (
        sales.groupby(["purchase_weekday", "purchase_hour"])["order_id"].count()
        .unstack(fill_value=0)
        .reindex(index=WEEKDAYS, columns=range(24), fill_value=0)
        .rename(columns=lambda hour: f"{hour:02d}h")
    )
    fig_heat = go.Figure(go.Heatmap(
        z=heat.values, x=heat.columns, y=heat.index,
        colorscale=COLORS["sequential"], xgap=2, ygap=2,
        colorbar=dict(title="Orders", thickness=10, tickfont_color=COLORS["muted"]),
        hovertemplate="%{y} %{x}<br>%{z:,} orders<extra></extra>",
    ))
    fig_heat.update_yaxes(autorange="reversed", showgrid=False)
    fig_heat.update_xaxes(dtick=3)

    left, right = st.columns(2)
    with left:
        chart_card("Average order value by month", "Mean of product + freight per order",
                   fig_aov, monthly_aov.assign(purchase_month=monthly_aov["purchase_month"].dt.strftime("%b %Y")).round(2))
    with right:
        chart_card("When customers buy", "Orders by weekday and hour of purchase (Brasília time)",
                   fig_heat, heat.reset_index(names="weekday"))

    # Seller concentration
    seller_revenue = sale_items.groupby("seller_id")["item_value"].sum().sort_values(ascending=False)
    seller_count = len(seller_revenue)
    if seller_count == 0:
        return
    seller_share = np.arange(1, seller_count + 1) / seller_count
    revenue_share = seller_revenue.cumsum().to_numpy() / seller_revenue.sum()

    fig_pareto = go.Figure(go.Scatter(
        x=seller_share, y=revenue_share, mode="lines", line=dict(color=COLORS["series"][0], width=2),
        hovertemplate="Top %{x:.0%} of sellers<br>%{y:.1%} of revenue<extra></extra>",
    ))
    thresholds = []
    for share in [0.01, 0.05, 0.10, 0.20, 0.50]:
        index = max(int(np.ceil(share * seller_count)) - 1, 0)
        thresholds.append({"top_sellers": f"Top {share:.0%}", "sellers": index + 1,
                           "revenue_share": round(revenue_share[index], 3)})
        if share in (0.10, 0.20):
            fig_pareto.add_scatter(x=[seller_share[index]], y=[revenue_share[index]], mode="markers",
                                   marker=dict(size=9, color=COLORS["series"][0],
                                               line=dict(width=2, color=COLORS["surface"])),
                                   hoverinfo="skip", showlegend=False)
            fig_pareto.add_annotation(x=seller_share[index], y=revenue_share[index],
                                      text=f"Top {share:.0%} → {revenue_share[index]:.0%} of revenue",
                                      showarrow=False, xanchor="left", xshift=12, yshift=-16,
                                      font=dict(color=COLORS["text"]))
    fig_pareto.update_layout(showlegend=False, hovermode="closest")
    fig_pareto.update_xaxes(tickformat=".0%", title_text="Share of sellers (largest first)", range=[0, 1])
    fig_pareto.update_yaxes(tickformat=".0%", title_text="Cumulative share of revenue", range=[0, 1.02])

    reviews_by_seller = orders.groupby("main_seller_id").agg(
        avg_score=("review_score", "mean"), late_rate=("is_late", "mean"))
    top_sellers = (
        sale_items.groupby("seller_id")
        .agg(seller_state=("seller_state", "first"), revenue=("item_value", "sum"),
             orders=("order_id", "nunique"))
        .join(reviews_by_seller)
        .sort_values("revenue", ascending=False)
        .head(15)
        .reset_index()
    )

    left, right = st.columns(2)
    with left:
        chart_card("Seller concentration", f"How revenue is spread across {seller_count:,} active sellers",
                   fig_pareto, pd.DataFrame(thresholds))
    with right:
        with st.container(border=True):
            st.markdown("**Top 15 sellers by revenue**")
            st.caption("Review score and late rate cover orders where the seller sold the main item")
            st.dataframe(
                top_sellers, hide_index=True, width="stretch", height=422,
                column_config={
                    "seller_id": "Seller ID",
                    "seller_state": "State",
                    "revenue": st.column_config.NumberColumn("Revenue (R$)", format="localized"),
                    "orders": st.column_config.NumberColumn("Orders", format="localized"),
                    "avg_score": st.column_config.NumberColumn("Avg score", format="%.2f ★"),
                    "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
                },
            )


def logistics_tab(orders: pd.DataFrame, states: pd.DataFrame, now: dict, before: dict | None) -> None:
    delivered = orders[orders["is_delivered"]]
    sales = orders[orders["is_sale"]]

    tiles = st.columns(6)
    kpi(tiles[0], "Delivered orders", "delivered", now, before, count)
    kpi(tiles[1], "Avg delivery time", "delivery_days", now, before, days, lower_is_better=True,
        help="Purchase to delivery at the customer's door")
    kpi(tiles[2], "Avg promised time", "estimated_days", now, before, days, lower_is_better=True,
        help="Purchase to the delivery date shown at checkout")
    kpi(tiles[3], "Late delivery rate", "late_rate", now, before, percent, delta="points", lower_is_better=True)
    kpi(tiles[4], "Freight share", "freight_share", now, before, percent, delta="points", lower_is_better=True,
        help="Freight as a share of the order total, averaged over orders")
    kpi(tiles[5], "Interstate orders", "interstate_rate", now, before, percent, delta="points",
        help="Orders where the main seller is in a different state from the customer")

    if delivered.empty:
        st.info("No delivered orders in this selection.")
        return

    # Monthly delivery performance
    monthly = (
        delivered.groupby("purchase_month")
        .agg(actual_days=("delivery_days", "mean"), promised_days=("estimated_days", "mean"),
             late_rate=("is_late", "mean"), delivered=("order_id", "count"))
        .reset_index()
    )
    fig_times = go.Figure()
    for column, name, color in [("promised_days", "Promised", COLORS["series"][1]),
                                ("actual_days", "Actual", COLORS["series"][0])]:
        fig_times.add_scatter(
            x=monthly["purchase_month"], y=monthly[column], name=name, mode=line_mode(len(monthly)),
            line=dict(color=color, width=2), marker=dict(size=8),
            hovertemplate=f"{name}: %{{y:.1f}} days<extra></extra>",
        )
        fig_times.add_annotation(x=monthly["purchase_month"].iloc[-1], y=monthly[column].iloc[-1], text=name,
                                 showarrow=False, xanchor="left", xshift=8, font=dict(color=COLORS["text"]))
    fig_times.update_layout(hovermode="x unified")
    fig_times.update_xaxes(tickformat="%b\n%Y", hoverformat="%b %Y")
    fig_times.update_yaxes(title_text="Days from purchase", rangemode="tozero")

    fig_late = monthly_line(monthly, "late_rate", "%{y:.1%} late", COLORS["series"][0])
    fig_late.update_yaxes(tickformat=".0%", rangemode="tozero")
    peak = monthly.loc[monthly["late_rate"].idxmax()]
    if len(monthly) > 2:
        fig_late.add_annotation(x=peak["purchase_month"], y=peak["late_rate"],
                                text=f"Peak {peak['purchase_month']:%b %Y}: {peak['late_rate']:.0%}",
                                showarrow=False, yshift=12, font=dict(color=COLORS["text"]))

    monthly_table = monthly.assign(purchase_month=monthly["purchase_month"].dt.strftime("%b %Y")).round(3)
    left, right = st.columns(2)
    with left:
        chart_card("Promised vs actual delivery time", "Average days from purchase, by purchase month",
                   fig_times, monthly_table)
    with right:
        chart_card("Late delivery rate by month", "Delivered after the promised date",
                   fig_late, monthly_table)

    # Geography
    state_names = dict(zip(states["state"], states["state_name"]))
    by_state = (
        delivered.groupby("customer_state")
        .agg(orders=("order_id", "count"), avg_delivery_days=("delivery_days", "mean"),
             late_rate=("is_late", "mean"), avg_freight=("freight_value", "mean"))
        .reset_index()
        .merge(states, left_on="customer_state", right_on="state")
    )
    fig_map = px.scatter_geo(
        by_state, lat="lat", lon="lng", size="orders", color="avg_delivery_days", size_max=38,
        color_continuous_scale=COLORS["sequential"], hover_name="state_name",
        custom_data=["orders", "avg_delivery_days", "late_rate"],
    )
    fig_map.update_traces(
        marker=dict(line=dict(width=2, color=COLORS["surface"]), opacity=0.9),
        hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]:,} delivered orders"
                      "<br>%{customdata[1]:.1f} days on average<br>%{customdata[2]:.1%} late<extra></extra>",
    )
    fig_map.update_geos(
        projection_type="mercator", lataxis_range=[-34, 6], lonaxis_range=[-75, -33],
        showland=True, landcolor=COLORS["land"], showcountries=True, countrycolor=COLORS["axis"],
        showcoastlines=True, coastlinecolor=COLORS["axis"], showocean=False, showlakes=False,
        showframe=False, bgcolor="rgba(0,0,0,0)",
    )
    fig_map.update_layout(coloraxis_colorbar=dict(title="Avg days", thickness=10, len=0.7,
                                                  tickfont_color=COLORS["muted"]))

    overall_late = delivered["is_late"].mean()
    by_state_sorted = by_state.sort_values("late_rate")
    fig_state_late = go.Figure()
    for above, name, color in [(False, "At or below overall rate", COLORS["neutral"]),
                               (True, "Above overall rate", COLORS["late"])]:
        part = by_state_sorted[(by_state_sorted["late_rate"] > overall_late) == above]
        fig_state_late.add_bar(
            x=part["late_rate"], y=part["customer_state"], orientation="h", name=name, marker_color=color,
            customdata=part[["state_name", "orders"]],
            hovertemplate="%{customdata[0]}<br>%{x:.1%} late · %{customdata[1]:,} orders<extra></extra>",
        )
    fig_state_late.update_yaxes(categoryorder="array", categoryarray=by_state_sorted["customer_state"].tolist())
    fig_state_late.update_xaxes(tickformat=".0%")
    fig_state_late.add_vline(x=overall_late, line_width=1, line_color=COLORS["text_secondary"])

    state_table = by_state[["customer_state", "state_name", "region", "orders", "avg_delivery_days",
                            "late_rate", "avg_freight"]].sort_values("orders", ascending=False).round(3)
    left, right = st.columns(2)
    with left:
        chart_card("Delivery time across Brazil", "Bubble size = delivered orders · color = average delivery days",
                   fig_map, state_table, height=max(440, 18 * len(by_state) + 40))
    with right:
        chart_card("Late delivery rate by customer state",
                   f"Line marks the overall rate ({percent(overall_late)}); states above it are highlighted",
                   fig_state_late, state_table, height=max(440, 18 * len(by_state) + 40), horizontal=True)

    # Where the time goes, and the delivery-time distribution
    stages = (
        delivered.groupby("customer_region")[list(STAGES)].mean()
        .reindex(REGIONS).dropna(how="all")
    )
    stages = stages.loc[stages.sum(axis=1).sort_values().index]
    fig_stages = go.Figure()
    for (column, name), color in zip(STAGES.items(), COLORS["series"]):
        fig_stages.add_bar(
            x=stages[column], y=stages.index, name=name, orientation="h", marker_color=color,
            marker_line=dict(width=2, color=COLORS["surface"]),
            text=[f"{value:.1f} d" if value >= 1.5 else "" for value in stages[column]],
            textposition="inside", insidetextanchor="middle", textfont=dict(color=ink_on(color)),
            hovertemplate=f"%{{y}}<br>{name}: %{{x:.1f}} days<extra></extra>",
        )
    fig_stages.update_layout(barmode="stack", legend_traceorder="normal")
    fig_stages.update_xaxes(title_text="Average days")

    clipped = delivered["delivery_days"].clip(upper=60)
    fig_hist = go.Figure(go.Histogram(
        x=clipped, xbins=dict(start=0, end=61, size=1), marker_color=COLORS["series"][0],
        marker_line=dict(width=1, color=COLORS["surface"]),
        hovertemplate="%{x} days<br>%{y:,} orders<extra></extra>",
    ))
    median_actual = delivered["delivery_days"].median()
    median_promised = delivered["estimated_days"].median()
    for value, label, shift in [(median_actual, "Median actual", 18), (median_promised, "Median promised", 36)]:
        fig_hist.add_vline(x=value, line_width=1, line_color=COLORS["text_secondary"])
        fig_hist.add_annotation(x=value, y=1, yref="paper", text=f"{label}: {value:.0f} d", showarrow=False,
                                xanchor="left", xshift=4, yshift=-shift, font=dict(color=COLORS["text"]))
    fig_hist.update_layout(showlegend=False, bargap=0)
    fig_hist.update_xaxes(title_text="Days from purchase to delivery (60+ grouped at 60)")
    distribution = clipped.round().value_counts().sort_index().rename_axis("delivery_days").reset_index(name="orders")

    left, right = st.columns(2)
    with left:
        chart_card("Where delivery time goes", "Average days per fulfillment stage, by customer region",
                   fig_stages, stages.round(2).reset_index(names="region").rename(columns=STAGES))
    with right:
        chart_card("Delivery time distribution", "Number of orders by delivery days; most arrive well before the promised date",
                   fig_hist, distribution)

    # Freight and routes
    freight = (
        sales.groupby("customer_region")
        .agg(freight_share=("freight_share", "mean"), avg_freight=("freight_value", "mean"))
        .reindex(REGIONS).dropna(how="all").sort_values("freight_share")
        .reset_index()
    )
    fig_freight = go.Figure(go.Bar(
        x=freight["freight_share"], y=freight["customer_region"], orientation="h", width=0.45,
        marker_color=COLORS["series"][0], customdata=freight[["avg_freight"]],
        hovertemplate="%{y}<br>%{x:.1%} of order value · avg R$ %{customdata[0]:.2f}<extra></extra>",
    ))
    fig_freight.update_xaxes(tickformat=".0%")

    routes = (
        delivered.dropna(subset=["seller_state"])
        .groupby(["seller_state", "customer_state"])
        .agg(orders=("order_id", "count"), avg_delivery_days=("delivery_days", "mean"),
             late_rate=("is_late", "mean"), avg_freight=("freight_value", "mean"))
        .sort_values("orders", ascending=False)
        .head(15)
        .reset_index()
    )
    left, right = st.columns(2)
    with left:
        chart_card("Freight cost by customer region", "Freight as a share of the order total, averaged over orders",
                   fig_freight, freight.round(3), height=300, horizontal=True)
    with right:
        with st.container(border=True):
            st.markdown("**Busiest seller → customer routes**")
            st.caption("Top 15 state pairs by delivered orders")
            st.dataframe(
                routes, hide_index=True, width="stretch", height=300,
                column_config={
                    "seller_state": "From (seller)",
                    "customer_state": "To (customer)",
                    "orders": st.column_config.NumberColumn("Orders", format="localized"),
                    "avg_delivery_days": st.column_config.NumberColumn("Avg days", format="%.1f"),
                    "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
                    "avg_freight": st.column_config.NumberColumn("Avg freight (R$)", format="%.2f"),
                },
            )


def customer_tab(orders: pd.DataFrame, states: pd.DataFrame, now: dict, before: dict | None) -> None:
    reviewed = orders[orders["review_score"].notna()]

    tiles = st.columns(6)
    kpi(tiles[0], "Avg review score", "score", now, before, stars, delta="absolute")
    kpi(tiles[1], "5-star reviews", "five_star_rate", now, before, percent, delta="points")
    kpi(tiles[2], "1–2 star reviews", "low_score_rate", now, before, percent, delta="points", lower_is_better=True)
    kpi(tiles[3], "Reviews with comment", "comment_rate", now, before, percent, delta="points")
    kpi(tiles[4], "Days to answer survey", "response_days", now, before, days, lower_is_better=True)
    kpi(tiles[5], "Repeat customers", "repeat_rate", now, before, percent, delta="points",
        help="Customers with more than one order in the selected period")

    if reviewed.empty:
        st.info("No reviews in this selection.")
        return

    # Score distribution and trend
    distribution = (
        reviewed["review_score"].astype(int).value_counts()
        .reindex(range(1, 6), fill_value=0).rename_axis("score").reset_index(name="reviews")
    )
    distribution["share"] = distribution["reviews"] / distribution["reviews"].sum()
    fig_distribution = go.Figure(go.Bar(
        x=[f"{score} ★" for score in distribution["score"]], y=distribution["share"],
        marker_color=[COLORS["scores"][score] for score in distribution["score"]],
        customdata=distribution["reviews"],
        hovertemplate="%{x}<br>%{y:.1%} · %{customdata:,} reviews<extra></extra>",
    ))
    fig_distribution.update_layout(showlegend=False)
    fig_distribution.update_yaxes(tickformat=".0%")

    monthly = (
        reviewed.groupby("purchase_month")
        .agg(avg_score=("review_score", "mean"), reviews=("order_id", "count"))
        .reset_index()
    )
    monthly = monthly[monthly["reviews"] >= 30]
    fig_trend = monthly_line(monthly, "avg_score", "%{y:.2f} ★", COLORS["series"][0])
    fig_trend.update_yaxes(range=[1, 5], dtick=1)

    left, right = st.columns(2)
    with left:
        chart_card("Review score distribution", "Share of reviews by star rating", fig_distribution,
                   distribution.round(3))
    with right:
        chart_card("Average review score by month", "By purchase month (months with 30+ reviews)", fig_trend,
                   monthly.assign(purchase_month=monthly["purchase_month"].dt.strftime("%b %Y")).round(2))

    # Score mix by delivery outcome
    outcome = np.select(
        [reviewed["is_late"] == 1, reviewed["is_late"] == 0], ["Late", "On time"], default="Not delivered")
    mix = (
        pd.crosstab(outcome, reviewed["review_score"].astype(int), normalize="index")
        .reindex(index=["Not delivered", "Late", "On time"], columns=range(1, 6), fill_value=0)
        .dropna(how="all")
    )
    mix = mix[mix.sum(axis=1) > 0]
    fig_mix = go.Figure()
    for score in range(1, 6):
        color = COLORS["scores"][score]
        fig_mix.add_bar(
            x=mix[score], y=mix.index, name=f"{score} ★", orientation="h", marker_color=color,
            marker_line=dict(width=2, color=COLORS["surface"]),
            text=[f"{value:.0%}" if value >= 0.07 else "" for value in mix[score]],
            textposition="inside", insidetextanchor="middle", textfont=dict(color=ink_on(color)),
            hovertemplate=f"%{{y}}<br>{score} ★: %{{x:.1%}}<extra></extra>",
        )
    fig_mix.update_layout(barmode="stack", legend_traceorder="normal")
    fig_mix.update_xaxes(tickformat=".0%", range=[0, 1])

    # States: late rate vs score
    state_names = dict(zip(states["state"], states["state_name"]))
    by_state = (
        reviewed.groupby("customer_state")
        .agg(late_rate=("is_late", "mean"), avg_score=("review_score", "mean"), reviews=("order_id", "count"))
        .reset_index()
    )
    by_state = by_state[by_state["reviews"] >= 30]
    by_state["state_name"] = by_state["customer_state"].map(state_names)
    # Label only the largest states and the outliers; hover names the rest.
    largest = by_state["reviews"].rank(ascending=False) <= 6
    outlier = (by_state["late_rate"] > 0.12) | (by_state["avg_score"] < 3.8)
    label_states = by_state["customer_state"].where(largest | outlier, "")
    sizes = 8 + 32 * np.sqrt(by_state["reviews"] / by_state["reviews"].max()) if not by_state.empty else []
    fig_states = go.Figure(go.Scatter(
        x=by_state["late_rate"], y=by_state["avg_score"], mode="markers+text", name="State",
        text=label_states, textposition="top center", textfont=dict(color=COLORS["text_secondary"]),
        marker=dict(size=sizes, color=COLORS["series"][0], opacity=0.85,
                    line=dict(width=2, color=COLORS["surface"])),
        customdata=by_state[["state_name", "reviews"]],
        hovertemplate="%{customdata[0]}<br>%{x:.1%} late · %{y:.2f} ★<br>%{customdata[1]:,} reviews<extra></extra>",
    ))
    if len(by_state) >= 3:
        slope, intercept = np.polyfit(by_state["late_rate"], by_state["avg_score"], 1, w=np.sqrt(by_state["reviews"]))
        x_range = np.array([by_state["late_rate"].min(), by_state["late_rate"].max()])
        fig_states.add_scatter(x=x_range, y=slope * x_range + intercept, mode="lines", name="Trend",
                               line=dict(color=COLORS["muted"], width=2), hoverinfo="skip")
    fig_states.update_layout(hovermode="closest")
    fig_states.update_xaxes(tickformat=".0%", title_text="Late delivery rate", showgrid=True,
                            gridcolor=COLORS["grid"])
    fig_states.update_yaxes(title_text="Average review score")

    left, right = st.columns(2)
    with left:
        chart_card("Score mix by delivery outcome", "Late orders get far more 1-star reviews",
                   fig_mix, mix.round(3).rename(columns=lambda score: f"{score} star").reset_index(names="outcome"),
                   height=260, horizontal=True)
    with right:
        chart_card("States with more late deliveries score lower",
                   "Each bubble is a customer state (30+ reviews); size = number of reviews",
                   fig_states, by_state.round(3), height=360)

    # Category ratings
    by_category = (
        reviewed[reviewed["main_category"] != "No items"].groupby("main_category")
        .agg(avg_score=("review_score", "mean"), reviews=("order_id", "count"), late_rate=("is_late", "mean"))
        .reset_index()
    )
    min_reviews = 100 if (by_category["reviews"] >= 100).sum() >= 5 else 20
    by_category = by_category[by_category["reviews"] >= min_reviews].sort_values("avg_score")
    lowest = by_category.head(15).iloc[::-1]
    overall = reviewed["review_score"].mean()
    fig_categories = go.Figure(go.Scatter(
        x=lowest["avg_score"], y=lowest["main_category"], mode="markers",
        marker=dict(size=10, color=COLORS["series"][0], line=dict(width=2, color=COLORS["surface"])),
        customdata=lowest[["reviews", "late_rate"]],
        hovertemplate="%{y}<br>%{x:.2f} ★ · %{customdata[0]:,} reviews · %{customdata[1]:.1%} late<extra></extra>",
    ))
    fig_categories.add_vline(x=overall, line_width=1, line_color=COLORS["text_secondary"])
    fig_categories.add_annotation(x=overall, y=1, yref="paper", text=f"Overall {overall:.2f} ★", showarrow=False,
                                  xanchor="left", xshift=4, yanchor="bottom", font=dict(color=COLORS["text"]))
    fig_categories.update_layout(showlegend=False)
    fig_categories.update_xaxes(title_text="Average review score")

    chart_card("Lowest-rated product categories",
               f"Categories with {min_reviews}+ reviews; the table view lists every category",
               fig_categories, by_category.round(3), height=420, horizontal=True, margin_top=28)


def explorer_tab(orders: pd.DataFrame, items: pd.DataFrame) -> None:
    st.caption("Details on demand: narrow the orders below, download them, or look up a single order.")
    status_col, outcome_col, score_col = st.columns(3)
    statuses = status_col.multiselect("Order status", sorted(orders["order_status"].unique()), placeholder="All")
    outcome = outcome_col.segmented_control("Delivery", ["All", "On time", "Late", "Not delivered"], default="All")
    scores = score_col.multiselect("Review score", [1, 2, 3, 4, 5], placeholder="All",
                                   format_func=lambda score: f"{score} ★")

    selected = orders
    if statuses:
        selected = selected[selected["order_status"].isin(statuses)]
    if outcome == "On time":
        selected = selected[selected["is_late"] == 0]
    elif outcome == "Late":
        selected = selected[selected["is_late"] == 1]
    elif outcome == "Not delivered":
        selected = selected[~selected["is_delivered"]]
    if scores:
        selected = selected[selected["review_score"].isin(scores)]

    columns = ["order_id", "purchase_ts", "customer_city", "customer_state", "main_category", "item_count",
               "order_value", "order_status", "delivery_days", "delay_days", "review_score"]
    table = selected.sort_values("purchase_ts", ascending=False)[columns]

    shown = table.head(5_000)
    st.markdown(f"**{len(table):,} orders** match" + (" (showing the 5,000 most recent)" if len(table) > 5_000 else ""))
    st.dataframe(
        shown, hide_index=True, width="stretch", height=420,
        column_config={
            "order_id": "Order ID",
            "purchase_ts": st.column_config.DatetimeColumn("Purchased", format="YYYY-MM-DD HH:mm"),
            "customer_city": "City",
            "customer_state": "State",
            "main_category": "Main category",
            "item_count": st.column_config.NumberColumn("Items", format="%d"),
            "order_value": st.column_config.NumberColumn("Value (R$)", format="%.2f"),
            "order_status": "Status",
            "delivery_days": st.column_config.NumberColumn("Delivery days", format="%.1f"),
            "delay_days": st.column_config.NumberColumn("Days vs promise", format="%+d",
                                                        help="Negative = early, positive = late"),
            "review_score": st.column_config.NumberColumn("Score", format="%d ★"),
        },
    )
    st.download_button("Download matching orders (CSV)", to_csv(table), file_name="olist_orders_selection.csv",
                       mime="text/csv", icon=":material/download:")

    st.divider()
    order_id = st.text_input("Look up an order", placeholder="Paste an order ID, e.g. from the table above").strip()
    if not order_id:
        return
    match = orders[orders["order_id"] == order_id]
    if match.empty:
        st.warning(f"No order with ID {order_id} in the current filters.")
        return
    order = match.iloc[0]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Order value", brl(order["order_value"]), border=True)
    summary_cols[1].metric("Status", order["order_status"].capitalize(), border=True)
    summary_cols[2].metric("Days vs promise", "–" if pd.isna(order["delay_days"]) else f"{order['delay_days']:+.0f}",
                           border=True)
    summary_cols[3].metric("Review score", "–" if pd.isna(order["review_score"]) else f"{order['review_score']:.0f} ★",
                           border=True)

    timeline = pd.DataFrame({
        "step": ["Purchased", "Payment approved", "Handed to carrier", "Delivered", "Promised by"],
        "date": [order["purchase_ts"], order["approved_ts"], order["carrier_ts"], order["delivered_ts"],
                 order["estimated_ts"]],
    })
    order_items = items.loc[items["order_id"] == order_id,
                            ["order_item_id", "category", "seller_id", "seller_state", "price", "freight_value"]]
    left, right = st.columns([2, 3])
    left.markdown(f"**Timeline** · {order['customer_city'].title()}, {order['customer_state']}")
    left.dataframe(timeline, hide_index=True, width="stretch",
                   column_config={"step": "Step", "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm")})
    right.markdown("**Items**")
    right.dataframe(order_items, hide_index=True, width="stretch")


@st.cache_data
def to_csv(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    missing = [name for name in PROCESSED_FILES if not (PROCESSED_DIR / name).exists()]
    if missing:
        st.error(f"Missing processed data: {', '.join(missing)} in {PROCESSED_DIR}. "
                 "Run data_preparation.ipynb first.")
        st.stop()

    orders, items, states = load_data()

    with st.sidebar:
        st.markdown("### About this dashboard")
        st.markdown(
            "Real, anonymized orders from **Olist Store**, a Brazilian marketplace (Kaggle), "
            "Jan 2017 – Aug 2018. Covers three departments: **Sales and Marketplace Operations**, "
            "**Logistics and Fulfillment**, and **Customer Experience**."
        )
        st.markdown("### Definitions")
        st.markdown(
            "- **Revenue**: product price + freight, excluding canceled and unavailable orders\n"
            "- **Late**: delivered after the date promised at checkout\n"
            "- **Main category / seller**: from the most expensive item in the order\n"
            "- **Deltas** on tiles compare with the previous period of the same length\n"
            "- **Region / state** refer to the customer's location"
        )
        st.caption("Every chart has a *Table view* with the exact numbers.")

    st.title("Olist Business Dashboard")
    current, previous, categories, period_label = filter_bar(orders, states)

    if current.empty:
        st.warning("No orders match these filters. Widen the period or clear a filter.")
        st.stop()

    current_items = items[items["order_id"].isin(current["order_id"])]
    if categories:
        current_items = current_items[current_items["category"].isin(categories)]

    now = summarize(current)
    before = summarize(previous) if previous is not None and not previous.empty else None
    st.caption(f"{period_label} · {len(current):,} orders"
               + ("" if before else " · no earlier period of the same length to compare with"))

    overview, sales, logistics, customers, explorer = st.tabs([
        "Overview", "Sales & Marketplace", "Logistics & Fulfillment", "Customer Experience", "Order Explorer",
    ])
    with overview:
        overview_tab(current, now, before)
    with sales:
        sales_tab(current, current_items, states, now, before)
    with logistics:
        logistics_tab(current, states, now, before)
    with customers:
        customer_tab(current, states, now, before)
    with explorer:
        explorer_tab(current, current_items)


main()
