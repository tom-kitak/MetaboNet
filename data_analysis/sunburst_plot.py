import pandas as pd
import plotly.express as px


# Creating the dataframe
data = {
    "metabolite": [
        "metab1",
        "metab2",
        "metab3",
        "metab4",
        "metab5",
        "metab6",
        "metab7",
    ],
    "subpathway": ["sub1", "sub1", "sub2", "sub4", "sub5", "sub4", "sub4"],
    "superpathway": [
        "super1",
        "super1",
        "super1",
        "super2",
        "super2",
        "super2",
        "super2",
    ],
    "metabolite_relative_importance": [0.1, 0.5, 0.2, 0.05, 0.02, 0.03, 0.1],
}

df = pd.DataFrame(data)

df["prediction"] = "Prediction"
fig = px.sunburst(
    df,
    path=["prediction", "superpathway", "subpathway", "metabolite"],
    values="metabolite_relative_importance",
    title="Super-pathway, sub-pathway and metabolite relative importance",
)
fig.show()
