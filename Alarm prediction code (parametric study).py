# ====================== ALARM PREDICTION CODE: PARAMETRIC STUDY ======================
# Changing time window and n(steps)
from numpy import array
import pandas as pd
from numpy import hstack
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
    end_ix = i + n_steps
    if end_ix > len(sequences):
      break
    seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)


# Import dataset (used for train, val, test)
df = pd.read_csv(r"dataset.csv")


# Preprocessing
df = df.rename(columns={'dT(CP)': 'dT'})
df['type'] = df['type'].replace('alarm', '1')
df['type'] = df['type'].replace('normal', '0')
df['type'] = df['type'].replace('override', '0')

def create_splitted_df(df, n_steps):
  historical_df = df[:-20]

  dmparent = historical_df['dmparent'].to_numpy()
  dmchild = historical_df['dmchild'].to_numpy()
  pt4 = historical_df['pt4'].to_numpy()
  pt3 = historical_df['pt3'].to_numpy()
  delivPumpActuation = historical_df['delivPumpActuation'].to_numpy()
  pt5 = historical_df['pt5'].to_numpy()
  foamDetResult = historical_df['foamDetResult'].to_numpy()
  tmp = historical_df['tmp'].to_numpy()
  diffFlow = historical_df['diffFlow'].to_numpy()
  bld = historical_df['bld'].to_numpy()
  encoderDelivPump = historical_df['encoderDelivPump'].to_numpy()
  delFlow = historical_df['delFlow'].to_numpy()
  currentWeightLoss = historical_df['currentWeightLoss'].to_numpy()
  arterial_revolve = historical_df['arterial_revolve'].to_numpy()
  ufPressureActuation = historical_df['ufPressureActuation'].to_numpy()
  condDO = historical_df['condDO'].to_numpy()
  condDOnf = historical_df['condDOnf'].to_numpy()
  pt6 = historical_df['pt6'].to_numpy()
  encoderUFPump = historical_df['encoderUFPump'].to_numpy()
  Ctot = historical_df['Ctot'].to_numpy()
  infusion_revolve = historical_df['infusion_revolve'].to_numpy()
  venous_revolve = historical_df['venous_revolve'].to_numpy()
  bmparent = historical_df['bmparent'].to_numpy()
  bmchild = historical_df['bmchild'].to_numpy()
  condTot = historical_df['condTot'].to_numpy()
  condTotnf = historical_df['condTotnf'].to_numpy()
  pVenous = historical_df['pVenous'].to_numpy()
  pPreFilt = historical_df['pPreFilt'].to_numpy()
  cond2 = historical_df['cond2'].to_numpy()
  ts1 = historical_df['ts1'].to_numpy()
  pPostInt = historical_df['pPostInt'].to_numpy()
  SecondStepPumpSpeed = historical_df['SecondStepPumpSpeed'].to_numpy()
  pt8 = historical_df['pt8'].to_numpy()
  encoder1 = historical_df['encoder1'].to_numpy()
  pArterial = historical_df['pArterial'].to_numpy()

  alarm_type = df['type'].to_numpy()[20:]

  dmparent = dmparent.reshape((len(dmparent), 1))
  dmchild = dmchild.reshape((len(dmchild), 1))
  pt4 = pt4.reshape((len(pt4), 1))
  pt3 = pt3.reshape((len(pt3), 1))
  delivPumpActuation = delivPumpActuation.reshape((len(delivPumpActuation), 1))
  pt5 = pt5.reshape((len(pt5), 1))
  foamDetResult = foamDetResult.reshape((len(foamDetResult), 1))
  tmp = tmp.reshape((len(tmp), 1))
  diffFlow = diffFlow.reshape((len(diffFlow), 1))
  bld = bld.reshape((len(bld), 1))
  encoderDelivPump = encoderDelivPump.reshape((len(encoderDelivPump), 1))
  delFlow = delFlow.reshape((len(delFlow), 1))
  currentWeightLoss = currentWeightLoss.reshape((len(currentWeightLoss), 1))
  arterial_revolve = arterial_revolve.reshape((len(arterial_revolve), 1))
  ufPressureActuation = ufPressureActuation.reshape((len(ufPressureActuation), 1))
  condDO = condDO.reshape((len(condDO), 1))
  condDOnf = condDOnf.reshape((len(condDOnf), 1))
  pt6 = pt6.reshape((len(pt6), 1))
  encoderUFPump = encoderUFPump.reshape((len(encoderUFPump), 1))
  Ctot = Ctot.reshape((len(Ctot), 1))
  infusion_revolve = infusion_revolve.reshape((len(infusion_revolve), 1))
  venous_revolve = venous_revolve.reshape((len(venous_revolve), 1))
  bmparent = bmparent.reshape((len(bmparent), 1))
  bmchild = bmchild.reshape((len(bmchild), 1))
  condTot = condTot.reshape((len(condTot), 1))
  condTotnf = condTotnf.reshape((len(condTotnf), 1))
  pVenous = pVenous.reshape((len(pVenous), 1))
  pPreFilt = pPreFilt.reshape((len(pPreFilt), 1))
  cond2 = cond2.reshape((len(cond2), 1))
  ts1 = ts1.reshape((len(ts1), 1))
  pPostInt = pPostInt.reshape((len(pPostInt), 1))
  SecondStepPumpSpeed = SecondStepPumpSpeed.reshape((len(SecondStepPumpSpeed), 1))
  pt8 = pt8.reshape((len(pt8), 1))
  encoder1 = encoder1.reshape((len(encoder1), 1))
  pArterial = pArterial.reshape((len(pArterial), 1))
  alarm_type = alarm_type.reshape((len(alarm_type), 1))

  dataset = hstack((dmparent, dmchild, pt4, pt3, delivPumpActuation, pt5, foamDetResult, tmp, diffFlow,
                    bld, encoderDelivPump, delFlow, currentWeightLoss,
                    arterial_revolve, ufPressureActuation, condDO, condDOnf, pt6, encoderUFPump,
                    Ctot, infusion_revolve, venous_revolve, bmparent, bmchild, condTot, condTotnf, pVenous,
                    pPreFilt, cond2, ts1, pPostInt, SecondStepPumpSpeed, pt8, encoder1, pArterial, alarm_type))

  X, y = split_sequences(dataset, n_steps)
  return X, y

X_all, y_all = create_splitted_df(df, 30)
y_all = y_all.astype(int).astype(str)


# Train/Validation/Test split
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)  # 0.1765*0.85 ≈ 0.15


# Reshape
nsamples, nx, ny = X_train.shape
df_train_dataset = X_train.reshape((nsamples, nx * ny))
nsamples2, nx2, ny2 = X_val.shape
df_val_dataset = X_val.reshape((nsamples2, nx2 * ny2))
nsamples3, nx3, ny3 = X_test.shape
df_test_dataset = X_test.reshape((nsamples3, nx3 * ny3))


# Model
LR_T100 = RandomForestClassifier(
    n_estimators=200,          # Più alberi -> più robustezza
    max_depth=30,              # Maggiore profondità, ma non infinita
    min_samples_split=3,       # Più sensibile a piccole variazioni
    min_samples_leaf=4,        # Foglie più piccole consentono più adattamento
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)


# Training
LR_T100.fit(df_train_dataset, y_train)
print("Random Forest Score on Training set -> ", LR_T100.score(df_train_dataset, y_train))


# Confusion matrix
pred_train = LR_T100.predict(df_train_dataset)
cm = confusion_matrix(y_train, pred_train)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Train Confusion Matrix', fontsize=16)
plt.show()

from sklearn.metrics import f1_score
f1score = f1_score(y_train, pred_train, pos_label='1')
print("F1-Score on Training set -> ", f1score)


# Validation set
print("Random Forest Classifier Score on Validation set -> ", LR_T100.score(df_val_dataset, y_val))
pred_val = LR_T100.predict(df_val_dataset)
cm2 = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Validation Confusion Matrix', fontsize=16)
plt.show()
f1score2 = f1_score(y_val, pred_val, pos_label='1')
print("F1-Score on Validation set -> ", f1score2)


# Test set
print("Random Forest Classifier Score on Test set -> ", LR_T100.score(df_test_dataset, y_test))
pred_test = LR_T100.predict(df_test_dataset)
cm3 = confusion_matrix(y_test, pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm3, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Test Confusion Matrix', fontsize=16)
plt.show()
f1score3 = f1_score(y_test, pred_test, pos_label='1')
print("F1-Score on Test set -> ", f1score3)


# Plot
importances = LR_T100.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(df_train_dataset.shape[1]), importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()