# Quality Escalation Intelligence System

## Overview

The **Quality Escalation Intelligence System** is a Streamlit-based analytics application for monitoring quality issues, estimating escalation risk, and supporting corrective-action planning.

This project is designed around a practical operational question:

> Which quality issues are most likely to escalate, and what should be reviewed first?

Rather than building a broad dashboard with every possible analytics module, this project focuses deliberately on escalation risk, issue prioritization, and quality governance visibility.

---

## Why This Project Exists

Quality workflows often involve many issue types, owners, severity levels, channels, and response timelines. When issue volume grows, it becomes difficult to identify which cases require immediate attention.

This application provides a structured way to:

- classify issue risk,
- identify high-risk quality cases,
- analyze escalation drivers,
- estimate escalation probability,
- and generate corrective-action recommendations.

---

## Current Capabilities

### Quality Issue Upload and Demo Data

The app supports:

- CSV upload for custom quality issue datasets,
- synthetic demo data for testing and portfolio demonstration,
- date-window filtering,
- owner-team filtering.

### Escalation Risk Queue

The dashboard ranks issues by an **Escalation Risk Score**, helping users focus on the highest-priority cases first.

### Quality Risk Distribution

The app visualizes how issues are distributed across risk bands, making it easier to understand overall quality pressure.

### Risk Driver Analysis

The system analyzes issue types by:

- issue volume,
- average escalation risk,
- escalation rate.

This supports review of recurring or high-impact issue categories.

### Escalation Prediction Model

A Random Forest classification model is used to estimate escalation behavior when enough labeled data is available.

The model provides:

- validation ROC-AUC,
- feature importance,
- simple interpretability for escalation drivers.

### Corrective-Action Recommendations

The recommendation layer translates risk signals into operational next steps such as reviewing recurring issues, aging issues, and high-impact issue clusters.

---

## Design Choice

This project intentionally emphasizes **risk prioritization and escalation intelligence** rather than forecasting or generic anomaly detection.

That is deliberate.

The goal is not to predict every future issue. The goal is to help quality reviewers identify which current issues deserve attention before they become operational escalations.

---

## Technology Stack

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-Learn
- Random Forest Classifier

---

## Example Use Cases

- Quality issue triage
- Escalation monitoring
- Review queue prioritization
- Corrective-action planning
- Operational quality governance
- Process improvement reporting

---

## Recommended Dataset Columns

The app works best with fields similar to:

- issue date
- issue type
- severity
- channel
- owner team
- age in days
- repeat issue flag
- customer impact
- escalation status

If some fields are missing, the synthetic/demo workflow can still demonstrate the application logic.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Enhancements

Possible improvements include:

- XGBoost escalation prediction,
- SHAP-based explainability,
- issue aging heatmaps,
- action-owner tracking,
- escalation timeline analysis,
- severity calibration logic,
- integration with workflow or ticketing exports.

---

## Portfolio Positioning

This project demonstrates applied analytics for **quality systems, escalation management, operational governance, and decision support**.

It is intentionally scoped around one strong decision problem:

> Which quality issues should be reviewed first, and why?
