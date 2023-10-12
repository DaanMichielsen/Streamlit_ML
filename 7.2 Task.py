import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, roc_curve, r2_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO, BytesIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from PIL import Image

def encode_data(df):
    label_encoder = LabelEncoder()
    yes_no_cols = ['Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring',
                'Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','class']
    for col in yes_no_cols:
        df[col] = label_encoder.fit_transform(df[col])

    df_encoded = pd.get_dummies(df, columns=['Gender'], prefix=['Gender'])
    df_encoded['Gender_Female'] = df_encoded['Gender_Female'].astype(int)
    df_encoded['Gender_Male'] = df_encoded['Gender_Male'].astype(int)
    df_encoded['Age'] = (df_encoded['Age'] - df_encoded['Age'].min()) / (df_encoded['Age'].max() - df_encoded['Age'].min())
    return df_encoded

# st.set_page_config(layout='wide')
st.title("Machine learning algorithm benchmark :robot_face::gear:")
st.caption('''
Created by [Daan Michielsen](https://github.com/DaanMichielsen)
           ''')

st.header("The dataset:bookmark_tabs:", divider='violet')

st.markdown("### [Early stage diabetes risk prediction dataset.](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) \nI chose this dataset because I can grasp the contents and I understand what I have to predict. It also complies with the minimum requirements for a dataset. I chose to perform **classification**.")

df = pd.read_csv('early_stage_diabetes_risk_prediction_dataset/diabetes_data_upload.csv', delimiter=',')
records_shown = 10
records_shows_options = [10, 20, 30, 40, 50]  # options for the selectbox

# Display the selectbox and update the number of shown records
c, c2, c3, c4, c5, c6 = st.columns(6)
with c:
    records_shown = st.selectbox(label=f"**Shown records:**", options=records_shows_options, key="df")
st.dataframe(df[:records_shown+1], height=None)
st.markdown("These are the columns in the dataset(17):")
st.markdown("- Age\n- Gender\n- Polyuria\n- Polydipsia\n- sudden weight loss\n- weakness\n- Polyphagia\n- Genital thrush\n- visual blurring\n- Itching\n- Irritability\n- delayed healing\n- partial paresis\n- muscle stiffness\n- Alopecia\n- Obesity\n- class")

st.header("The algorithms:robot_face:", divider='violet')
st.markdown("I compared 1 baseline algorithm to 2 new algorithms I have not used before. For the **baseline** I chose the **decision tree algorithm**. For the other 2 algorithms I went with **SVC** and **AdaBoostClassifier**.")
st.markdown("In order to train the data I had to perform some cleaning, encoding and normalizing which left me with this result:")
df_encoded = encode_data(df)

records_shown_encoded = 10  # default number of records to show
records_shows_options_encoded = [10, 20, 30, 40, 50]  # options for the select slider

# Display the slider and update the number of shown records
c, c2, c3, c4, c5, c6 = st.columns(6)
with c:
    records_shown_encoded = st.selectbox(label=f"**Shown records:**", options=records_shows_options_encoded, key="df_encode")
st.dataframe(df_encoded[:records_shown_encoded+1])
st.header("The benchmark:male-detective:", divider='violet')

st.markdown("If you just want to walk through the benchmark I suggest you run the code but feel free to play around with the parameters to see the change in result.")

feature_cols = ['Age','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring',
               'Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','Gender_Female', 'Gender_Male']
X = df_encoded[feature_cols]
y = df_encoded['class']

# Define the test size options
test_size_options = [10, 20, 30, 40, 50]

# Display the slider and update the test size
test_size = st.slider('Select test size:', min_value=10, max_value=50, step=10, value=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

test_size_text = "**Ratio:** %" + str(int((100-test_size))) + ' Training,  %' + str(int(test_size)) + ' Testing'

st.markdown(test_size_text)

# Define the data for the bar chart
train_size = len(X_train)
test_size = len(X_test)

with st.expander(f"X test({test_size} records)"):
    st.subheader("X test")
    st.dataframe(X_test)
with st.expander(f"X train({train_size} records)"):
    st.subheader("X train")
    st.dataframe(X_train)

st.markdown("For the algorithms we can **configure some parameters** which impact the outcome of the model. For the **decision tree** we can change the **max_depth**, for the **SVC** algorithm we can select a **kernel**. For our application of binary classification the *linear kernel* will probably be the best. For the *adaptive boost* algorithm we can change *n estimator* which is the amount of times it will run the *base estimator*.")
st.markdown("### parameter tuning:gear:")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### :blue[Decision tree]")
    max_depth = st.slider('Select max depth:', min_value=1, max_value=10, step=1, value=10, help="Spoiler, the tree does not go past 10 hence 10 as max input")
with col2:
    st.markdown("### :green[SVC]")
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = st.selectbox(label="Select kernel:", options=kernel_options, index=0, help="Usually there is a *precomputed* kernel but this requires a square dataset which we don't have")
with col3:
    st.markdown("### :red[AdaBoost]")
    n_estimators = st.number_input(label="Select amount of estimators:", min_value=10, max_value=250, step=10, value=50, help="The mount of weak learners the algorithm will use to become better")

if st.button(":white[Fit model]", type='secondary', use_container_width=True):
    with st.spinner('Training model, predicting and evaluating results...'):
        st.header("Results/Metrics:bar_chart:", divider='violet')
        st.subheader(f"Parameters:")
        st.markdown(f'max depth=<span style="color:blue">{max_depth}</span> | kernel=<span style="color:green">{kernel}</span> | estimators=<span style="color:red">{n_estimators}</span>', unsafe_allow_html=True)
        clf = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=max_depth)
        svc = SVC(random_state=42, kernel=kernel)
        base_estimator = DecisionTreeClassifier(max_depth=1)
        adabc = AdaBoostClassifier(random_state=42, base_estimator=base_estimator, n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        svc.fit(X_train, y_train)
        adabc.fit(X_train, y_train)

        dot_data = StringIO()
        export_graphviz(clf, out_file = dot_data, filled = True, rounded = True,
                        special_characters = True, feature_names = feature_cols, class_names=['Negative','Positive'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        image = Image.open(BytesIO(graph.create_png()))
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Decision Tree","Prediction vs Test","Overall Metrics","Confusion Matrix","ROC Curve","Precision-Recall Curve"])
        with tab1:
            st.markdown("### Decision Tree")
            st.image(image=image)
            st.caption(f"Image of decision tree with max depth of {max_depth}")
        with tab2:
            st.subheader("Prediction results")
            y_pred_decision_tree = clf.predict(X_test)
            y_pred_svc = svc.predict(X_test)
            y_pred_adaboost = adabc.predict(X_test)

            # Count the occurrences of actual values in y_test
            actual_counts = np.bincount(y_test)

            # Count the occurrences of predicted values for each algorithm
            predicted_counts_decision_tree = np.bincount(y_pred_decision_tree)
            predicted_counts_svc = np.bincount(y_pred_svc)
            predicted_counts_adaboost = np.bincount(y_pred_adaboost)

            # Create labels for the classes
            labels = np.arange(len(actual_counts))

            # Width of each bar in the bar chart
            width = 0.2

            # Create a bar chart with 4 bars for each class
            fig, ax = plt.subplots()
            bar1 = ax.bar(labels - width, actual_counts, width, label='Actual', color='orange')
            bar2 = ax.bar(labels, predicted_counts_decision_tree, width, label='Decision Tree', color='blue')
            bar3 = ax.bar(labels + width, predicted_counts_svc, width, label='SVC', color='green')
            bar4 = ax.bar(labels + 2 * width, predicted_counts_adaboost, width, label='AdaBoost', color='red')

            # Set labels and legend
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Actual vs. Predicted Values')

            # Set the x-axis tick labels
            ax.set_xticks(labels)
            ax.set_xticklabels(['Positive', 'Negative'])

            # Legend
            ax.legend()

            # Add labels to the bars
            for bar_ in [bar1, bar2, bar3, bar4]:
                for rect in bar_:
                    height = rect.get_height()
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            # Display the bar chart
            st.pyplot(fig, use_container_width=True)
            st.caption(f"Bar chart comparing the prediction vs. the real values")

        decision_metrics = []
        svc_metrics = []
        adaboost_metrics = []
        # Decision Tree Classifier
        # R2 Score
        r2_decision_tree = r2_score(y_test, y_pred_decision_tree)
        decision_metrics.append(r2_decision_tree)
        # Accuracy
        accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
        decision_metrics.append(accuracy_decision_tree)
        # Precision
        precision_decision_tree = precision_score(y_test, y_pred_decision_tree, average='weighted')
        decision_metrics.append(precision_decision_tree)
        # Recall
        recall_decision_tree = recall_score(y_test, y_pred_decision_tree, average='weighted')
        decision_metrics.append(recall_decision_tree)
        # F1-score
        f1_score_decision_tree = f1_score(y_test, y_pred_decision_tree, average='weighted')
        decision_metrics.append(recall_decision_tree)
        # Confusion Matrix
        confusion_matrix_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
        # SVC Classifier
        # R2 Score
        r2_svc = r2_score(y_test, y_pred_svc)
        svc_metrics.append(r2_svc)
        # Accuracy
        accuracy_svc = accuracy_score(y_test, y_pred_svc)
        svc_metrics.append(accuracy_svc)
        # Precision
        precision_svc = precision_score(y_test, y_pred_svc, average='weighted')
        svc_metrics.append(precision_svc)
        # Recall
        recall_svc = recall_score(y_test, y_pred_svc, average='weighted')
        svc_metrics.append(recall_svc)
        # F1-score
        f1_score_svc = f1_score(y_test, y_pred_svc, average='weighted')
        svc_metrics.append(f1_score_svc)
        # Confusion Matrix
        confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)
        # AdaBoost Classifier
        # R2 Score
        r2_adaboost = r2_score(y_test, y_pred_adaboost)
        adaboost_metrics.append(r2_adaboost)
        # Accuracy
        accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
        adaboost_metrics.append(accuracy_adaboost)
        # Precision
        precision_adaboost = precision_score(y_test, y_pred_adaboost, average='weighted')
        adaboost_metrics.append(precision_adaboost)
        # Recall
        recall_adaboost = recall_score(y_test, y_pred_adaboost, average='weighted')
        adaboost_metrics.append(recall_adaboost)
        # F1-score
        f1_score_adaboost = f1_score(y_test, y_pred_adaboost, average='weighted')
        adaboost_metrics.append(f1_score_adaboost)
        # Confusion Matrix
        confusion_matrix_adaboost = confusion_matrix(y_test, y_pred_adaboost)

        
        decision_metrics = [r2_decision_tree, accuracy_decision_tree, precision_decision_tree, recall_decision_tree, f1_score_decision_tree]
        svc_metrics = [r2_svc, accuracy_svc, precision_svc, recall_svc, f1_score_svc]
        adaboost_metrics = [r2_adaboost, accuracy_adaboost, precision_adaboost, recall_adaboost, f1_score_adaboost]

        # Define the algorithm names
        algorithm_names = ["Decision Tree", "SVC", "AdaBoost"]

        # Create a dictionary of metrics for each algorithm
        metrics_dict = {
            "Decision Tree": decision_metrics,
            "SVC": svc_metrics,
            "AdaBoost": adaboost_metrics
        }

        # Create a pandas dataframe from the metrics dictionary
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["R2 Score", "Accuracy", "Precision", "Recall", "F1-score"]).transpose()
        with tab3:
            st.subheader("Overall Metrics")
            # Display the dataframe
            st.dataframe(metrics_df, use_container_width=True)
        with tab4:
            st.subheader("Confusion Matrix")
            # Define the confusion matrices for the three classifiers
            confusion_matrices = [confusion_matrix_decision_tree, confusion_matrix_svc, confusion_matrix_adaboost]

            # Class names (you may need to adjust these based on your specific dataset)
            class_names = ["Positive", "Negative"]

            # Titles for each confusion matrix
            titles = ["Decision Tree Classifier", "SVC Classifier", "AdaBoost Classifier"]

            # Plot confusion matrices side by side
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            for i in range(3):
                sns.set(font_scale=1.5)
                # Compute confusion matrix
                cm = confusion_matrices[i]
                # Convert confusion matrix to pandas dataframe
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="g", ax=axs[i])
                axs[i].set_title(titles[i], fontsize=20)
                axs[i].set_xlabel('Predicted label', fontsize=16)
                axs[i].set_ylabel('True label', fontsize=16)

            plt.tight_layout()

            # Display the confusion matrices
            st.pyplot(fig)
            st.caption(f"Confusion matrixes for each algorithm")
        with tab5:
            st.subheader("ROC Curve")
            classifiers = [clf, svc, adabc]
            predictions = [y_pred_decision_tree, y_pred_svc, y_pred_adaboost]
            classifier_names = ["Decision Tree", "SVC", "AdaBoost"]

            # Set up the plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Define a color cycle for different classifiers
            colors = cycle(['b', 'g', 'r'])

            # Plot ROC curve and calculate AUC for each classifier
            for classifier, prediction, color, name in zip(classifiers, predictions, colors, classifier_names):
                fpr, tpr, _ = roc_curve(y_test, prediction)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

            # Set labels and legend
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc="lower right")

            # Display the ROC curves
            st.pyplot(fig)
            st.caption(f"ROC curve for each algoritm", help="A ROC curve is a plot of the true positive rate (y-axis) versus the false positive rate (x-axis) for different classification thresholds, and it shows the tradeoff between sensitivity and specificity for a binary classification model. The closer the curve is to the top left corner of the plot, the better the model does at classifying the data into categories.")
        with tab6:
            st.subheader("Precision-Recall Curve")
            precision_tree, recall_tree, thresholds = precision_recall_curve(y_test, y_pred_decision_tree)
            precision_svc, recall_svc, thresholds = precision_recall_curve(y_test, y_pred_svc)
            precision_adaboost, recall_adaboost, thresholds = precision_recall_curve(y_test, y_pred_adaboost)

            # Plot the precision-recall curve
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(recall_tree, precision_tree, color='blue')
            ax.plot(recall_svc, precision_svc, color='green')
            ax.plot(recall_adaboost, precision_adaboost, color='red')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(['Decision tree','SVC', 'AdaBoost'])
            st.pyplot(fig)
            st.caption(f"Precision-recall curve for each algorithm", help="A precision-recall curve is a plot of precision (y-axis) versus recall (x-axis) for different classification thresholds. It shows how changing the classification threshold affects the tradeoff between false positives and false negatives. A high area under the curve represents both high recall and high precision, indicating a model that returns accurate results for the majority of classes it selects.")