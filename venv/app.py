import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd 
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('C:/DSS/venv/FinalProjectDSS.csv', sep=',')

df ['y'] = df ['y'].map({'no' : 0, 'yes': 1})
    
df2_columns = ['age','day','duration','pdays','previous','y']   
selected_data = df[df2_columns]

# Clean and preprocess the data
df.dropna(inplace=True)  # Remove missing values
# Convert target variable to binary

# Exploratory data analysis (EDA)
fig1 = px.histogram(selected_data, x='age', color='y', barmode='group', nbins=20, title='Age Distribution by Outcome')

fig2 = px.box(selected_data, x='y', y='duration', title='Duration by Outcome')

# Feature engineering
selected_data['recent_pdays'] = selected_data['pdays'].apply(lambda x: 1 if x == -1 else 0)  # Create a binary feature for recent contact

# Model selection
X = selected_data[['age', 'duration', 'pdays', 'previous']]
y = selected_data['y']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate model performance
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
y_pred = lr.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

fig3 = px.imshow(conf_matrix, labels=dict(x="Predicted", y="True", color="Count"), x=['No', 'Yes'], y=['No', 'Yes'], title='Confusion Matrix')
fig4 = px.histogram(selected_data, x='duration', color='y', nbins=20, barmode='group', histnorm='percent', title='Duration Distribution by Outcome')
fig5 = px.histogram(selected_data, x='previous', color='y', nbins=10, barmode='group', histnorm='percent', title='Previous Distribution by Outcome')


descriptive_job = dcc.Graph(
                    id='y-graph'
                )

descriptive_job_drop_menu = dcc.Dropdown(
                    id='job-dropdown',
                    options=[{'label': i, 'value': i} for i in df['job'].unique()],
                    value='management'
                )
 
predictive_heatmap = dcc.Graph(
                    id ='heatmap-graph'
                )

# Create the dashboard
app = dash.Dash(__name__)
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3(children=[descriptive_job_drop_menu, descriptive_job])),
        dbc.Col(html.H3(children=[predictive_heatmap])),
    ]),
    html.H1('Diagnostic Analytics Dashboard'),
    dcc.Graph(id='fig1', figure=fig1),
    dcc.Graph(id='fig2', figure=fig2),
    dcc.Graph(id='fig3', figure=fig3),
    dcc.Graph(id='fig4', figure=fig4),
    dcc.Graph(id='fig5', figure=fig5)
])

def update_y_graph(selected_job):
    
    df = pd.read_csv('C:/DSS/venv/FinalProjectDSS.csv', sep=',')

    df_selected = df[df['job'] == selected_job]
    y_count = df_selected['y'].value_counts()
    data1 = [
        {'x': ['Yes', 'No'], 'y': [y_count['yes'], y_count['no']], 'type': 'bar', 'name': 'y'},
    ]
    layout1 = go.Layout(
        title='Descriptive Analytics',
        xaxis=dict(title='y'),
        yaxis=dict(title='count')
    )
    return {'data': data1, 'layout': layout1}

def update_predictive(_):
    df = pd.read_csv('C:/DSS/venv/FinalProjectDSS.csv', sep=',')

    # Preprocess the data
    le = LabelEncoder()
    df.loc[:,'poutcome'] = le.fit_transform(df['poutcome'])
    df.loc[:,'loan'] = le.fit_transform(df['loan'])
    df.loc[:,'job'] = le.fit_transform(df['job'])
    df.loc[:,'marital'] = le.fit_transform(df['marital'])
    X = df.drop(['loan', 'y'], axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0][0]+cm[1][1])/sum(sum(cm))

    # Create confusion matrix bar chart
    labels = ['No', 'Yes']
    tp, fp, fn, tn = cm.ravel()
    values = [tn, fp, fn, tp]
    colors = ['rgb(228,26,28)', 'rgb(55,126,184)', 'rgb(228,26,28)', 'rgb(55,126,184)']
    trace = go.Bar(x=labels, y=values, marker=dict(color=colors))
    data = [trace]
    layout = go.Layout(title='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label'))
    fig = go.Figure(data=data, layout=layout)

    return fig

@app.callback(
    Output('y-graph', 'figure'),
    [Input('job-dropdown', 'value')])
def update_Descriptive(selected_job):
    layout1 = update_y_graph(selected_job)
    return layout1

@app.callback(Output('heatmap-graph', 'figure'),
              [Input('heatmap-graph', 'id')])
def update_Predictive_graph(_):
    fig = update_predictive(_)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
