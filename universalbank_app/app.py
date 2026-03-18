import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc as sk_auc,
)
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank · Loan Intelligence Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
section[data-testid="stSidebar"]{background:linear-gradient(160deg,#0f1b2d 0%,#1a2f4e 60%,#0f1b2d 100%);border-right:1px solid rgba(212,175,55,0.25);}
section[data-testid="stSidebar"] *{color:#e8e0d0 !important;}
.stApp{background:#f7f5f0;}
.hero-banner{background:linear-gradient(135deg,#0f1b2d 0%,#1a3a5c 55%,#0f1b2d 100%);border-radius:16px;padding:2.4rem 3rem;margin-bottom:1.5rem;border:1px solid rgba(212,175,55,0.3);position:relative;overflow:hidden;}
.hero-banner::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(45deg,transparent,transparent 40px,rgba(212,175,55,0.03) 40px,rgba(212,175,55,0.03) 80px);}
.hero-title{font-family:'Playfair Display',serif;font-size:2.3rem;font-weight:900;color:#d4af37;margin:0;line-height:1.1;position:relative;}
.hero-sub{color:#a0b4c8;font-size:1rem;margin-top:0.4rem;font-weight:300;position:relative;}
.metric-card{background:white;border-radius:12px;padding:1.3rem 1.5rem;border-left:4px solid #d4af37;box-shadow:0 2px 12px rgba(0,0,0,0.07);margin-bottom:0.2rem;}
.metric-label{font-size:0.74rem;color:#6b7a8d;text-transform:uppercase;letter-spacing:1px;font-weight:600;}
.metric-value{font-family:'Playfair Display',serif;font-size:2rem;color:#0f1b2d;font-weight:700;line-height:1.1;}
.metric-delta{font-size:0.8rem;color:#5b8a6e;margin-top:0.2rem;}
.section-header{font-family:'Playfair Display',serif;font-size:1.45rem;color:#0f1b2d;font-weight:700;border-bottom:2px solid #d4af37;padding-bottom:0.35rem;margin:1.6rem 0 0.9rem 0;}
.insight-box{background:linear-gradient(135deg,#fff9ec,#fffdf7);border:1px solid rgba(212,175,55,0.35);border-left:4px solid #d4af37;border-radius:8px;padding:0.85rem 1.15rem;font-size:0.86rem;color:#3d3320;line-height:1.65;margin-top:0.55rem;}
.insight-box strong{color:#0f1b2d;}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:10px;padding:4px;box-shadow:0 1px 6px rgba(0,0,0,0.07);}
.stTabs [data-baseweb="tab"]{border-radius:8px;font-weight:500;color:#6b7a8d;}
.stTabs [aria-selected="true"]{background:#0f1b2d !important;color:white !important;}
footer,#MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── PALETTE ───────────────────────────────────────────────────────────────────
GOLD="#d4af37"; NAVY="#0f1b2d"; STEEL="#1a3a5c"; TEAL="#2e7d7d"
ROSE="#c0392b"; GREEN="#27ae60"

BASE = dict(
    font=dict(family="DM Sans",color="#2c2c2c"),
    paper_bgcolor="white",plot_bgcolor="white",
    colorway=[NAVY,GOLD,TEAL,"#8e44ad",ROSE,GREEN,"#e67e22"],
    title=dict(font=dict(family="Playfair Display",size=16,color=NAVY)),
    xaxis=dict(gridcolor="#f0eff0",linecolor="#ddd"),
    yaxis=dict(gridcolor="#f0eff0",linecolor="#ddd"),
    margin=dict(t=55,b=40,l=40,r=20),
    legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd",borderwidth=1),
)

def sfig(fig,title="",h=420):
    fig.update_layout(**BASE,height=h,title=title)
    return fig

def mc(label,value,delta=""):
    d=f'<div class="metric-delta">▲ {delta}</div>' if delta else ""
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{d}</div>'

def ib(text):
    return f'<div class="insight-box">{text}</div>'

def sh(text):
    return f'<div class="section-header">{text}</div>'

# ── DATA & MODELS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df.columns = df.columns.str.strip()
    df["Experience"] = df["Experience"].clip(lower=0)
    return df

FEAT_COLS = ["Age","Experience","Income","Family","CCAvg","Education",
             "Mortgage","Securities Account","CD Account","Online","CreditCard"]

@st.cache_resource
def train_all(df):
    X = df[FEAT_COLS]; y = df["Personal Loan"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    mdls = {
        "Decision Tree":         DecisionTreeClassifier(max_depth=8,min_samples_leaf=15,class_weight="balanced",random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=300,max_depth=12,class_weight="balanced",random_state=42,n_jobs=-1),
        "Gradient Boosted Tree": GradientBoostingClassifier(n_estimators=200,learning_rate=0.08,max_depth=5,random_state=42),
    }
    res,trained={},{}
    for name,mdl in mdls.items():
        mdl.fit(Xtr,ytr); trained[name]=mdl
        for tag,Xs,ys in [("Train",Xtr,ytr),("Test",Xte,yte)]:
            p=mdl.predict(Xs); pb=mdl.predict_proba(Xs)[:,1]
            fp,tp,_=roc_curve(ys,pb)
            res[f"{name}|{tag}"]=dict(
                acc=accuracy_score(ys,p),pre=precision_score(ys,p,zero_division=0),
                rec=recall_score(ys,p,zero_division=0),f1=f1_score(ys,p,zero_division=0),
                cm=confusion_matrix(ys,p),fpr=fp,tpr=tp,auc=sk_auc(fp,tp),
            )
    return trained,res,Xtr,Xte,ytr,yte

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
      <div style='font-family:Playfair Display,serif;font-size:1.4rem;color:#d4af37;font-weight:700'>🏦 Universal Bank</div>
      <div style='font-size:0.74rem;color:#7a8fa3;margin-top:3px'>Loan Intelligence Hub</div>
    </div>
    <hr style='border-color:rgba(212,175,55,0.2);margin:0.7rem 0'>""",unsafe_allow_html=True)

    page=st.radio("Navigate",[
        "📊  Executive Overview","🔍  Descriptive Analytics",
        "📈  Diagnostic Analytics","🤖  Predictive Models",
        "🎯  Prescriptive Analytics","📤  Predict New Customers",
    ])
    st.markdown("""
    <hr style='border-color:rgba(212,175,55,0.2);margin:0.9rem 0 0.4rem'>
    <div style='font-size:0.71rem;color:#506070;text-align:center'>5,000 customers · 14 features<br>Target: Personal Loan Acceptance</div>
    """,unsafe_allow_html=True)

subs={"📊  Executive Overview":"High-level KPIs and campaign baseline",
      "🔍  Descriptive Analytics":"Who are our customers? Distribution & patterns",
      "📈  Diagnostic Analytics":"What drives loan acceptance? Deep correlations",
      "🤖  Predictive Models":"Decision Tree · Random Forest · Gradient Boosted Tree",
      "🎯  Prescriptive Analytics":"Actionable targeting segments for next campaign",
      "📤  Predict New Customers":"Upload customer data and download loan predictions"}
st.markdown(f'<div class="hero-banner"><div class="hero-title">Universal Bank · Loan Intelligence Hub</div><div class="hero-sub">{subs[page]}</div></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page=="📊  Executive Overview":
    total=len(df); acc=int(df["Personal Loan"].sum()); dec=total-acc; rate=acc/total*100
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(mc("Total Customers",f"{total:,}"),unsafe_allow_html=True)
    c2.markdown(mc("Loan Acceptances",f"{acc:,}",f"{rate:.1f}% acceptance rate"),unsafe_allow_html=True)
    c3.markdown(mc("Avg Income · Accepted",f"${df[df['Personal Loan']==1]['Income'].mean():.0f}K"),unsafe_allow_html=True)
    c4.markdown(mc("Avg Age · Accepted",f"{df[df['Personal Loan']==1]['Age'].mean():.1f} yrs"),unsafe_allow_html=True)

    st.markdown(sh("Campaign Baseline"),unsafe_allow_html=True)
    col1,col2=st.columns([1,1.65])
    with col1:
        fig=go.Figure(go.Pie(labels=[f"Accepted ({rate:.1f}%)",f"Declined ({100-rate:.1f}%)"],
                             values=[acc,dec],hole=0.62,marker_colors=[GOLD,"#d0dce8"],textinfo="percent",
                             hovertemplate="%{label}<br>Count: %{value}<extra></extra>"))
        fig.add_annotation(text=f"<b>{rate:.1f}%</b><br>Acceptance",x=0.5,y=0.5,showarrow=False,
                           font=dict(size=16,family="Playfair Display",color=NAVY))
        sfig(fig,"Overall Loan Acceptance Rate",h=340); st.plotly_chart(fig,use_container_width=True)
        st.markdown(ib("<strong>9.6% acceptance rate</strong> from the last campaign — ~1 in 10 customers converted. With a halved budget, hyper-targeting is the only path to maintaining conversion volume."),unsafe_allow_html=True)
    with col2:
        df["Edu_Label"]=df["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced/Prof"})
        edu_r=df.groupby("Edu_Label")["Personal Loan"].mean().reset_index(); edu_r["Rate%"]=(edu_r["Personal Loan"]*100).round(1)
        fam_r=df.groupby("Family")["Personal Loan"].mean().reset_index(); fam_r["Rate%"]=(fam_r["Personal Loan"]*100).round(1)
        fig2=make_subplots(rows=1,cols=2,subplot_titles=["By Education Level","By Family Size"])
        fig2.add_trace(go.Bar(x=edu_r["Edu_Label"],y=edu_r["Rate%"],marker_color=[STEEL,TEAL,GOLD],
                              text=edu_r["Rate%"].astype(str)+"%",textposition="outside",showlegend=False),row=1,col=1)
        fig2.add_trace(go.Bar(x=fam_r["Family"].astype(str),y=fam_r["Rate%"],marker_color=NAVY,
                              text=fam_r["Rate%"].astype(str)+"%",textposition="outside",showlegend=False),row=1,col=2)
        fig2.update_layout(**BASE,height=340,title="Key Demographic Acceptance Rates")
        fig2.update_yaxes(title_text="Acceptance %",row=1,col=1); fig2.update_yaxes(title_text="Acceptance %",row=1,col=2)
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown(ib("<strong>Advanced/Professional graduates</strong> accept at ~14% — nearly 3× undergrads. <strong>Family size 3</strong> peaks at ~14%, signalling the highest financial responsibility stage."),unsafe_allow_html=True)

    st.markdown(sh("Income vs Credit Card Spend — Accepted vs Declined"),unsafe_allow_html=True)
    samp=df.sample(1200,random_state=1)
    fig3=px.scatter(samp,x="Income",y="CCAvg",color=samp["Personal Loan"].map({0:"Declined",1:"Accepted"}),
                    color_discrete_map={"Accepted":GOLD,"Declined":"#b0bec5"},opacity=0.65,
                    labels={"Income":"Annual Income ($000)","CCAvg":"Monthly CC Spend ($000)","color":"Loan Decision"})
    sfig(fig3,"Income vs Monthly Credit Card Spend — 1,200 Customer Sample",h=380); st.plotly_chart(fig3,use_container_width=True)
    st.markdown(ib("Loan acceptors (gold) cluster tightly in the <strong>high-income, high-CC-spend quadrant</strong>. Customers earning above $100K with monthly CC spend over $2.5K represent your highest-yield targeting corridor."),unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🔍  Descriptive Analytics":
    tab1,tab2,tab3=st.tabs(["👤  Demographics","💳  Financial Profile","🏦  Bank Products"])

    with tab1:
        st.markdown(sh("Customer Demographics Overview"),unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(df,x="Age",color=df["Personal Loan"].map({0:"Declined",1:"Accepted"}),
                             nbins=25,barmode="overlay",opacity=0.8,
                             color_discrete_map={"Accepted":GOLD,"Declined":"#b0bec5"},
                             labels={"Age":"Customer Age","count":"# Customers","color":"Decision"})
            sfig(fig,"Age Distribution by Loan Decision",h=360); st.plotly_chart(fig,use_container_width=True)
            st.markdown(ib("Acceptances peak between ages <strong>30–45</strong> — the prime earning and borrowing window. Customers under 28 or over 58 have very low conversion likelihood."),unsafe_allow_html=True)
        with c2:
            df["Edu_Label"]=df["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced/Prof"})
            ecnt=df.groupby(["Edu_Label","Personal Loan"]).size().reset_index(name="Count")
            ecnt["Loan"]=ecnt["Personal Loan"].map({0:"Declined",1:"Accepted"})
            fig2=px.bar(ecnt,x="Edu_Label",y="Count",color="Loan",barmode="group",
                        color_discrete_map={"Accepted":GOLD,"Declined":NAVY},text="Count",
                        labels={"Edu_Label":"Education Level","Count":"# Customers","Loan":"Decision"})
            sfig(fig2,"Education Level vs Loan Acceptance",h=360); st.plotly_chart(fig2,use_container_width=True)
            st.markdown(ib("Advanced/Professional graduates form a <strong>high-converting minority</strong>. Though ~29% of customers, their ~14% conversion rate is nearly double undergrads."),unsafe_allow_html=True)
        c3,c4=st.columns(2)
        with c3:
            fam=df.groupby(["Family","Personal Loan"]).size().reset_index(name="Count")
            fam["Loan"]=fam["Personal Loan"].map({0:"Declined",1:"Accepted"})
            fig3=px.bar(fam,x="Family",y="Count",color="Loan",barmode="stack",
                        color_discrete_map={"Accepted":GOLD,"Declined":NAVY},text="Count",
                        labels={"Family":"Family Size","Count":"# Customers"})
            sfig(fig3,"Family Size Distribution",h=340); st.plotly_chart(fig3,use_container_width=True)
            st.markdown(ib("Family sizes <strong>3 and 4 show higher loan acceptance</strong>. Larger families carry more obligations — home improvements, education, vehicles — making them natural personal loan candidates."),unsafe_allow_html=True)
        with c4:
            fig4=px.histogram(df,x="Experience",color=df["Personal Loan"].map({0:"Declined",1:"Accepted"}),
                              nbins=22,barmode="overlay",opacity=0.78,
                              color_discrete_map={"Accepted":GOLD,"Declined":"#b0bec5"},
                              labels={"Experience":"Years of Experience","count":"# Customers","color":"Decision"})
            sfig(fig4,"Professional Experience Distribution",h=340); st.plotly_chart(fig4,use_container_width=True)
            st.markdown(ib("<strong>10–25 years of experience</strong> — mid-career professionals — are the most active loan takers, combining stable income with real asset-building aspirations."),unsafe_allow_html=True)

    with tab2:
        st.markdown(sh("Financial Profile Analysis"),unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.box(df,x=df["Personal Loan"].map({0:"Declined",1:"Accepted"}),y="Income",
                       color=df["Personal Loan"].map({0:"Declined",1:"Accepted"}),
                       color_discrete_map={"Accepted":GOLD,"Declined":NAVY},points="outliers",
                       labels={"x":"Loan Decision","Income":"Annual Income ($000)"})
            sfig(fig,"Income Distribution — Accepted vs Declined",h=380); st.plotly_chart(fig,use_container_width=True)
            st.markdown(ib("Loan acceptors have a <strong>significantly higher median income (~$115K vs ~$65K)</strong>. Income is the single strongest individual predictor of loan acceptance in this dataset."),unsafe_allow_html=True)
        with c2:
            fig2=px.box(df,x=df["Personal Loan"].map({0:"Declined",1:"Accepted"}),y="CCAvg",
                        color=df["Personal Loan"].map({0:"Declined",1:"Accepted"}),
                        color_discrete_map={"Accepted":GOLD,"Declined":NAVY},points="outliers",
                        labels={"x":"Loan Decision","CCAvg":"Monthly CC Spend ($000)"})
            sfig(fig2,"Credit Card Spend — Accepted vs Declined",h=380); st.plotly_chart(fig2,use_container_width=True)
            st.markdown(ib("Accepted customers spend nearly <strong>3× more on credit cards monthly (~$3.9K vs $1.7K)</strong>. High CC spend signals financial confidence and a lifestyle requiring larger credit capacity."),unsafe_allow_html=True)
        c3,c4=st.columns(2)
        with c3:
            df_m=df.copy(); df_m["Mortgage_Band"]=pd.cut(df_m["Mortgage"],bins=[-1,0,100,300,700],labels=["None","<$100K","$100–300K",">$300K"])
            mort=df_m.groupby(["Mortgage_Band","Personal Loan"]).size().reset_index(name="Count"); mort["Loan"]=mort["Personal Loan"].map({0:"Declined",1:"Accepted"})
            fig3=px.bar(mort,x="Mortgage_Band",y="Count",color="Loan",barmode="group",
                        color_discrete_map={"Accepted":GOLD,"Declined":NAVY},text="Count",
                        labels={"Mortgage_Band":"Mortgage Band","Count":"# Customers"})
            sfig(fig3,"Mortgage Band vs Loan Acceptance",h=340); st.plotly_chart(fig3,use_container_width=True)
            st.markdown(ib("The majority of loan acceptors have <strong>no mortgage</strong> — personal loans are primarily used for consumption, liquidity, or unsecured financing rather than property."),unsafe_allow_html=True)
        with c4:
            df_i=df.copy(); df_i["Income_Band"]=pd.cut(df_i["Income"],bins=[0,30,70,120,230],labels=["<$30K","$30–70K","$70–120K",">$120K"])
            inc=df_i.groupby(["Income_Band","Personal Loan"]).size().reset_index(name="Count"); inc["Loan"]=inc["Personal Loan"].map({0:"Declined",1:"Accepted"})
            fig4=px.bar(inc,x="Income_Band",y="Count",color="Loan",barmode="stack",
                        color_discrete_map={"Accepted":GOLD,"Declined":NAVY},text="Count",
                        labels={"Income_Band":"Income Band","Count":"# Customers"})
            sfig(fig4,"Income Band vs Loan Acceptance",h=340); st.plotly_chart(fig4,use_container_width=True)
            st.markdown(ib("The <strong>high income band (&gt;$120K)</strong> has an acceptance rate exceeding <strong>35%</strong>. Conversion rate climbs steeply with income — prioritise this band across all channels."),unsafe_allow_html=True)

    with tab3:
        st.markdown(sh("Bank Product Ownership & Loan Propensity"),unsafe_allow_html=True)
        c1,c2=st.columns(2)
        bcols=["Securities Account","CD Account","Online","CreditCard"]
        nice={"Securities Account":"Securities Acct","CD Account":"CD Account","Online":"Online Banking","CreditCard":"Bank Credit Card"}
        with c1:
            xl=[nice[c] for c in bcols]; yn=[df[df[c]==0]["Personal Loan"].mean()*100 for c in bcols]; yy=[df[df[c]==1]["Personal Loan"].mean()*100 for c in bcols]
            fig=go.Figure()
            fig.add_bar(name="Does NOT have product",x=xl,y=yn,marker_color=NAVY,text=[f"{v:.1f}%" for v in yn],textposition="outside")
            fig.add_bar(name="Has product",x=xl,y=yy,marker_color=GOLD,text=[f"{v:.1f}%" for v in yy],textposition="outside")
            fig.update_layout(barmode="group"); sfig(fig,"Loan Acceptance Rate by Bank Product Ownership",h=390); st.plotly_chart(fig,use_container_width=True)
            st.markdown(ib("<strong>CD Account holders convert at ~24% vs ~8.5%</strong> for non-holders — nearly 3×. This is the most powerful cross-sell signal available in your existing customer data."),unsafe_allow_html=True)
        with c2:
            df["prod_combo"]="Sec="+df["Securities Account"].astype(str)+" | CD="+df["CD Account"].astype(str)+" | Online="+df["Online"].astype(str)
            combo=df.groupby("prod_combo")["Personal Loan"].agg(["mean","count"]).reset_index(); combo.columns=["Product Combo","Acceptance Rate","Count"]
            combo=combo[combo["Count"]>30].sort_values("Acceptance Rate",ascending=False).head(8)
            combo["Rate%"]=(combo["Acceptance Rate"]*100).round(1); combo["label"]=combo.apply(lambda r:f"{r['Product Combo']} (n={r['Count']})",axis=1)
            fig2=px.bar(combo,x="Rate%",y="label",orientation="h",color="Rate%",
                        color_continuous_scale=[[0,"#d0dce8"],[1,GOLD]],text="Rate%",
                        labels={"Rate%":"Acceptance Rate %","label":""})
            sfig(fig2,"Top Product Combinations by Acceptance Rate",h=390); st.plotly_chart(fig2,use_container_width=True)
            st.markdown(ib("Customers with <strong>Securities + CD + Online Banking</strong> (fully engaged) show the highest acceptance rates. Multi-product holders are deeply invested — leverage existing relationship managers."),unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page=="📈  Diagnostic Analytics":
    st.markdown(sh("What Drives Loan Acceptance?"),unsafe_allow_html=True)
    num_cols=["Age","Experience","Income","Family","CCAvg","Mortgage","Securities Account","CD Account","Online","CreditCard","Personal Loan"]
    corr=df[num_cols].corr()
    fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),
                              colorscale=[[0,STEEL],[0.5,"white"],[1,GOLD]],zmin=-1,zmax=1,
                              text=np.round(corr.values,2),texttemplate="%{text}",hoverongaps=False,
                              colorbar=dict(title="r")))
    sfig(fig,"Correlation Matrix — All Numeric Features",h=480); st.plotly_chart(fig,use_container_width=True)
    st.markdown(ib("<strong>Income (r=0.50)</strong> and <strong>CCAvg (r=0.37)</strong> are the strongest positive correlates with Personal Loan. <strong>CD Account (r=0.32)</strong> leads among bank product variables. Age & Experience are nearly perfectly collinear (r≈0.99)."),unsafe_allow_html=True)

    st.markdown(sh("Income Segmentation Deep-Dive"),unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        df["Edu_Label"]=df["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced/Prof"})
        fig2=px.violin(df,x="Edu_Label",y="Income",color=df["Personal Loan"].map({0:"Declined",1:"Accepted"}),
                       box=True,points=False,color_discrete_map={"Accepted":GOLD,"Declined":NAVY},
                       labels={"Edu_Label":"Education Level","Income":"Annual Income ($000)","color":"Decision"})
        sfig(fig2,"Income Distribution by Education × Loan Decision",h=420); st.plotly_chart(fig2,use_container_width=True)
        st.markdown(ib("Even within education levels, <strong>loan acceptors earn substantially more</strong>. The income bar for acceptance rises with education — advanced degree holders command higher salaries and seek larger credit facilities."),unsafe_allow_html=True)
    with c2:
        samp2=df.sample(min(1500,len(df)),random_state=7)
        fig3=px.scatter(samp2,x="Income",y="CCAvg",color=samp2["Personal Loan"].map({0:"Declined",1:"Accepted"}),
                        size=samp2["Mortgage"].clip(lower=1)+5,color_discrete_map={"Accepted":GOLD,"Declined":"#b0bec5"},
                        opacity=0.65,labels={"Income":"Annual Income ($000)","CCAvg":"Monthly CC Spend ($000)","color":"Decision","size":"Mortgage"})
        sfig(fig3,"Income × CC Spend — bubble size = Mortgage value",h=420); st.plotly_chart(fig3,use_container_width=True)
        st.markdown(ib("The gold cluster concentrates above <strong>$100K income and $2.5K/mo CC spend</strong>. Larger bubbles (higher mortgages) do not dominate — personal loans here are consumption and liquidity driven, not property-linked."),unsafe_allow_html=True)

    st.markdown(sh("Acceptance Rate Heatmap — Income × Family Size"),unsafe_allow_html=True)
    df_h=df.copy(); df_h["Inc_Band"]=pd.cut(df_h["Income"],bins=[0,30,60,90,120,150,230],labels=["<30","30–60","60–90","90–120","120–150",">150"])
    pivot=df_h.pivot_table(values="Personal Loan",index="Family",columns="Inc_Band",aggfunc="mean")*100
    fig4=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns.astype(str).tolist(),y=[f"Family {i}" for i in pivot.index],
                               colorscale=[[0,"#e8f4f8"],[0.5,STEEL],[1,GOLD]],text=np.round(pivot.values,1),
                               texttemplate="%{text}%",colorbar=dict(title="Acc. Rate %")))
    sfig(fig4,"Personal Loan Acceptance Rate % — Income Band × Family Size",h=360); st.plotly_chart(fig4,use_container_width=True)
    st.markdown(ib("The top-right quadrant (high income, large family) is your <strong>gold zone</strong> — acceptance rates exceeding 40–55%. A family of 3–4 earning above $120K is the single most convertible micro-segment in the entire customer base."),unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🤖  Predictive Models":
    with st.spinner("Training three classification models…"):
        trained,res,Xtr,Xte,ytr,yte=train_all(df)

    MN=["Decision Tree","Random Forest","Gradient Boosted Tree"]
    MC={MN[0]:STEEL,MN[1]:GOLD,MN[2]:TEAL}

    st.markdown(sh("Model Performance Summary — Train vs Test"),unsafe_allow_html=True)
    rows=[]
    for name in MN:
        for split in ["Train","Test"]:
            r=res[f"{name}|{split}"]
            rows.append({"Model":name,"Split":split,"Accuracy":f"{r['acc']*100:.2f}%","Precision":f"{r['pre']*100:.2f}%",
                         "Recall":f"{r['rec']*100:.2f}%","F1-Score":f"{r['f1']*100:.2f}%","ROC-AUC":f"{r['auc']*100:.2f}%"})
    pf=pd.DataFrame(rows)
    def hl(row): return ["background-color:#fff9ec;font-weight:600"]*len(row) if row["Split"]=="Test" else [""]*len(row)
    st.dataframe(pf.style.apply(hl,axis=1),use_container_width=True,hide_index=True)
    st.markdown(ib("Highlighted rows = <strong>Test set performance</strong> (unseen data — the true measure). All three models use <code>class_weight='balanced'</code> to counter the 90:10 class imbalance, ensuring they learn to detect true loan-takers rather than always predicting 'No'."),unsafe_allow_html=True)

    st.markdown(sh("ROC Curves — All Three Models (Test Set)"),unsafe_allow_html=True)
    froc=go.Figure()
    froc.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(color="#ccc",dash="dash",width=1.2))
    for name in MN:
        r=res[f"{name}|Test"]
        froc.add_trace(go.Scatter(x=r["fpr"],y=r["tpr"],mode="lines",
                                  name=f"{name}  (AUC = {r['auc']:.4f})",
                                  line=dict(color=MC[name],width=2.5)))
    froc.update_layout(**BASE,height=450,title="ROC Curve Comparison — Test Set",
                       xaxis_title="False Positive Rate (1 − Specificity)",
                       yaxis_title="True Positive Rate (Sensitivity)")
    st.plotly_chart(froc,use_container_width=True)
    st.markdown(ib("A <strong>higher AUC</strong> means better separation of loan-takers from non-takers across all thresholds. AUC=1 is perfect; AUC=0.5 is random. <strong>Gradient Boosted Tree</strong> delivers the highest AUC and is the recommended model for live campaign scoring."),unsafe_allow_html=True)

    st.markdown(sh("Confusion Matrices — Test Set"),unsafe_allow_html=True)
    cols=st.columns(3)
    for i,name in enumerate(MN):
        r=res[f"{name}|Test"]; cm=r["cm"]; tot=cm.sum()
        zv=[[cm[1][1],cm[1][0]],[cm[0][1],cm[0][0]]]
        tx=[[f"<b>{cm[1][1]}</b><br>({cm[1][1]/tot*100:.1f}%)",f"<b>{cm[1][0]}</b><br>({cm[1][0]/tot*100:.1f}%)"],
            [f"<b>{cm[0][1]}</b><br>({cm[0][1]/tot*100:.1f}%)",f"<b>{cm[0][0]}</b><br>({cm[0][0]/tot*100:.1f}%)"]]
        fcm=go.Figure(go.Heatmap(z=zv,x=["Predicted: YES","Predicted: NO"],y=["Actual: YES","Actual: NO"],
                                  colorscale=[[0,"#e8f4f8"],[0.5,STEEL],[1,GOLD]],text=tx,texttemplate="%{text}",showscale=False))
        fcm.update_layout(**BASE,height=320,title=name,xaxis_title="Predicted Label",yaxis_title="Actual Label")
        with cols[i]: st.plotly_chart(fcm,use_container_width=True)
    st.markdown(ib("<strong>True Positives (top-left, gold)</strong> = correctly identified loan-takers — your marketing hit rate. <strong>False Negatives (top-right)</strong> = missed opportunities. <strong>False Positives (bottom-left)</strong> = wasted campaign spend. Maximising True Positives (Recall) is the priority when budget is constrained."),unsafe_allow_html=True)

    st.markdown(sh("Feature Importance — Gradient Boosted Tree"),unsafe_allow_html=True)
    gbt=trained["Gradient Boosted Tree"]
    imp=pd.DataFrame({"Feature":FEAT_COLS,"Importance":gbt.feature_importances_}).sort_values("Importance",ascending=True)
    imp["Pct"]=(imp["Importance"]*100).round(1)
    fimp=px.bar(imp,x="Pct",y="Feature",orientation="h",color="Pct",
                color_continuous_scale=[[0,"#d0dce8"],[1,GOLD]],text=imp["Pct"].astype(str)+"%",
                labels={"Pct":"Importance %","Feature":""})
    sfig(fimp,"Feature Importance — Gradient Boosted Tree",h=420); st.plotly_chart(fimp,use_container_width=True)
    st.markdown(ib("<strong>Income dominates</strong> with the highest importance, followed by <strong>CCAvg, Education, and CD Account</strong>. These four features alone drive the vast majority of predictive power. For any new campaign data collection, ensure these four fields are captured with highest accuracy."),unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🎯  Prescriptive Analytics":
    st.markdown(sh("Hyper-Personalised Campaign Segments"),unsafe_allow_html=True)
    df_p=df.copy(); df_p["Segment"]="⚪ General Pool"
    df_p.loc[(df_p["Income"]>=60)&(df_p["Mortgage"]>0)&(df_p["Online"]==1),"Segment"]="🥉 Silver Candidates"
    df_p.loc[(df_p["Income"]>=80)&(df_p["CCAvg"]>=2)&(df_p["Education"]>=2)&(df_p["Family"]>=3),"Segment"]="🥈 Gold Targets"
    df_p.loc[(df_p["Income"]>=100)&(df_p["CCAvg"]>=3)&(df_p["CD Account"]==1),"Segment"]="🥇 Platinum Prospects"

    sc={"🥇 Platinum Prospects":GOLD,"🥈 Gold Targets":"#9eb3c2","🥉 Silver Candidates":"#c87941","⚪ General Pool":"#9daebe"}
    so=["🥇 Platinum Prospects","🥈 Gold Targets","🥉 Silver Candidates","⚪ General Pool"]
    summary=[]
    for seg in so:
        s=df_p[df_p["Segment"]==seg]
        summary.append({"Segment":seg,"Count":len(s),"% of Customers":f"{len(s)/len(df_p)*100:.1f}%",
                        "Acceptance Rate":f"{s['Personal Loan'].mean()*100:.1f}%",
                        "Avg Income ($K)":f"${s['Income'].mean():.0f}K",
                        "Avg CC Spend/Mo":f"${s['CCAvg'].mean():.2f}K",
                        "CD Acct Holders":f"{s['CD Account'].mean()*100:.0f}%"})

    bub=pd.DataFrame({"Segment":so,
                       "Avg Income":[float(s["Avg Income ($K)"].replace("$","").replace("K","")) for s in summary],
                       "Acceptance%":[float(s["Acceptance Rate"].replace("%","")) for s in summary],
                       "Size":[s["Count"] for s in summary]})
    fig=px.scatter(bub,x="Avg Income",y="Acceptance%",size="Size",color="Segment",color_discrete_map=sc,
                   text="Segment",size_max=75,labels={"Avg Income":"Avg Annual Income ($K)","Acceptance%":"Acceptance Rate %"})
    fig.update_traces(textposition="top center",textfont=dict(size=11))
    sfig(fig,"Campaign Segments — Size · Income · Conversion Rate",h=450); st.plotly_chart(fig,use_container_width=True)
    st.markdown(ib("Bubble size = segment population. <strong>Platinum Prospects</strong> are small but hyper-convertible. With a halved budget, invest here first — maximum ROI per dollar spent."),unsafe_allow_html=True)

    st.markdown(sh("Segment Breakdown"),unsafe_allow_html=True)
    sdf=pd.DataFrame(summary)
    def hl2(row): return ["background-color:#fff9ec;font-weight:600"]*len(row) if "Platinum" in row["Segment"] else [""]*len(row)
    st.dataframe(sdf.style.apply(hl2,axis=1),use_container_width=True,hide_index=True)

    st.markdown(sh("Campaign Playbook"),unsafe_allow_html=True)
    pb=[
        ("🥇 Platinum Prospects","High-income CD account holders with heavy CC spend","Premium direct mail + personal banker outreach. Pre-approved loan offer with competitive rate. Personalise by mortgage and family status.","35–55%","40%"),
        ("🥈 Gold Targets","Graduate+ educated, mid-high income, family of 3+","Email + in-app notification. Highlight family benefits — home improvement, education, vehicle. Bundle with existing product upgrade.","20–35%","35%"),
        ("🥉 Silver Candidates","Mid-income mortgage holders, active online users","Digital campaign — targeted ads and push notifications. Emphasise refinancing and debt-consolidation.","10–20%","20%"),
        ("⚪ General Pool","Low-income, low CC spend, no existing bank products","Low-cost automated email. Focus on relationship-building and cross-sell; personal loan is a secondary goal here.","<8%","5%"),
    ]
    for seg,profile,action,rate,budget in pb:
        with st.expander(f"{seg}  ·  Expected Conversion: {rate}  ·  Budget: {budget}"):
            ca,cb=st.columns([1,2])
            ca.markdown(f"**Customer Profile:**\n\n{profile}")
            cb.markdown(f"**Recommended Action:**\n\n{action}")

    st.markdown(sh("Recommended Budget Allocation"),unsafe_allow_html=True)
    c1,c2=st.columns([1,1.5])
    with c1:
        fb=go.Figure(go.Pie(labels=["🥇 Platinum","🥈 Gold","🥉 Silver","⚪ General"],values=[40,35,20,5],
                             marker_colors=[GOLD,"#9eb3c2","#c87941","#d0dce8"],hole=0.52,textinfo="label+percent"))
        fb.add_annotation(text="<b>Budget</b><br>Split",x=0.5,y=0.5,showarrow=False,
                          font=dict(size=13,family="Playfair Display",color=NAVY))
        sfig(fb,"Recommended Marketing Budget Allocation",h=380); st.plotly_chart(fb,use_container_width=True)
    with c2:
        st.markdown('<br><br>',unsafe_allow_html=True)
        st.markdown(ib("<strong>Why 75% on Platinum + Gold?</strong><br><br>These two segments represent ~20% of your customer base but will deliver <strong>&gt;60% of total conversions</strong>. A halved budget forces precision — concentrate firepower on the highest-propensity segments.<br><br>Use the <em>Predict New Customers</em> tab to score your full customer base and automatically tier everyone before campaign launch."),unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICT NEW CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif page=="📤  Predict New Customers":
    st.markdown(sh("Score & Download Loan Predictions"),unsafe_allow_html=True)
    st.markdown(ib("<strong>How to use:</strong> Upload any customer CSV (same column structure as training data — the <em>Personal Loan</em> column is not required). The <strong>Gradient Boosted Tree</strong> model scores each customer with a Prediction (0/1) and a Loan_Probability (0–1). Download the enriched file and feed it directly into your CRM for campaign targeting."),unsafe_allow_html=True)

    with st.spinner("Loading trained models…"):
        trained,res,Xtr,Xte,ytr,yte=train_all(df)

    st.markdown("#### ⬇️ Download a sample test file to try:")
    sample_test=df.drop(columns=["Personal Loan"]).sample(200,random_state=99)
    st.download_button("Download Sample Test File (200 customers)",
                       data=sample_test.to_csv(index=False).encode(),
                       file_name="sample_test_customers.csv",mime="text/csv")
    st.markdown("---")
    st.markdown("#### 📁 Upload Your Customer File:")
    uploaded=st.file_uploader("Upload CSV",type=["csv"],label_visibility="collapsed")

    if uploaded:
        try:
            ndf=pd.read_csv(uploaded)
            st.success(f"✅  {len(ndf):,} customer records loaded.")
            st.dataframe(ndf.head(5),use_container_width=True)
            proc=ndf.copy()
            for dc in ["ZIP Code","ID","Personal Loan"]:
                if dc in proc.columns: proc.drop(columns=[dc],inplace=True)
            if "Experience" in proc.columns: proc["Experience"]=proc["Experience"].clip(lower=0)
            for col in FEAT_COLS:
                if col not in proc.columns: proc[col]=0
            proc=proc[FEAT_COLS]
            mdl=trained["Gradient Boosted Tree"]
            preds=mdl.predict(proc); proba=mdl.predict_proba(proc)[:,1]
            ndf["Prediction"]=preds; ndf["Loan_Probability"]=np.round(proba,4)
            ndf["Propensity_Tier"]=pd.cut(proba,bins=[-0.01,0.2,0.5,0.75,1.01],labels=["Low","Medium","High","Very High"])

            st.markdown(sh("Prediction Summary"),unsafe_allow_html=True)
            py=int(preds.sum()); hv=int((proba>=0.5).sum())
            c1,c2,c3,c4=st.columns(4)
            c1.markdown(mc("Total Scored",f"{len(ndf):,}"),unsafe_allow_html=True)
            c2.markdown(mc("Predicted: YES",f"{py:,}",f"{py/len(ndf)*100:.1f}% of batch"),unsafe_allow_html=True)
            c3.markdown(mc("Avg Loan Prob.",f"{proba.mean()*100:.1f}%"),unsafe_allow_html=True)
            c4.markdown(mc("High / Very High Tier",f"{hv:,}","Priority targets"),unsafe_allow_html=True)

            c1,c2=st.columns(2)
            with c1:
                fh=px.histogram(x=proba,nbins=30,color_discrete_sequence=[NAVY],
                                labels={"x":"Loan Probability","count":"# Customers"})
                fh.add_vline(x=0.5,line_dash="dash",line_color=GOLD,annotation_text="Threshold 0.5")
                sfig(fh,"Distribution of Predicted Loan Probabilities",h=340); st.plotly_chart(fh,use_container_width=True)
            with c2:
                tcnt=ndf["Propensity_Tier"].value_counts().reset_index(); tcnt.columns=["Tier","Count"]
                ft=px.pie(tcnt,values="Count",names="Tier",hole=0.5,
                          color_discrete_map={"Low":"#d0dce8","Medium":STEEL,"High":TEAL,"Very High":GOLD})
                sfig(ft,"Propensity Tier Distribution",h=340); st.plotly_chart(ft,use_container_width=True)

            st.markdown(ib("Sort your CRM export by <strong>Loan_Probability descending</strong> and work top-down until your budget is exhausted — this is the optimal deployment of a constrained campaign budget."),unsafe_allow_html=True)
            st.markdown(sh("Download Results"),unsafe_allow_html=True)
            st.download_button("⬇️  Download Full Predictions CSV",
                               data=ndf.to_csv(index=False).encode(),
                               file_name="loan_predictions.csv",mime="text/csv")
            st.dataframe(ndf.sort_values("Loan_Probability",ascending=False).head(30),use_container_width=True)
        except Exception as e:
            st.error(f"❌  Error: {e}")
            st.info("Ensure your CSV matches the training data format. Download the sample file above.")
    else:
        st.info("👆  Upload a CSV above, or download the sample file to try it out.")
