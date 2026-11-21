import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

BASE = Path(__file__).resolve().parent
PREP = BASE / "ML_Preprocessing"
MODEL_DIR = BASE / "Model_Comparison"
ANOMALY_DIR = BASE / "Anomaly_Detection"

FEATURES = [
    "pH", "DO", "BOD", "Cond", "Nitrate", "COD", "TotalColi",
    "FecalColi", "Turbidity", "TotAlk", "Chloride", "Sulphate",
    "Hardness", "TDS", "TSS", "Fluoride"
]

BIS_LIMITS = {
    "pH": {"acceptable_min": 6.5, "acceptable_max": 8.5,
           "permissible_max": 8.5, "unit": ""},
    "Turbidity": {"acceptable_max": 1, "permissible_max": 5, "unit": "NTU"},
    "TDS": {"acceptable_max": 500, "permissible_max": 2000, "unit": "mg/L"},
    "Chloride": {"acceptable_max": 250, "permissible_max": 1000, "unit": "mg/L"},
    "Fluoride": {"acceptable_max": 1.0, "permissible_max": 1.5, "unit": "mg/L"},
    "Nitrate": {"acceptable_max": 45, "permissible_max": 45, "unit": "mg/L"},
    "Sulphate": {"acceptable_max": 200, "permissible_max": 400, "unit": "mg/L"},
    "Hardness": {"acceptable_max": 200, "permissible_max": 600, "unit": "mg/L as CaCO3"},
    "TotAlk": {"acceptable_max": 200, "permissible_max": 600, "unit": "mg/L as CaCO3"},
}

app = FastAPI(
    title="Water Quality Classification and Anomaly Detection API",
    description=(
        "API for water-quality classification, drinking-water screening "
        "and anomaly detection."
    ),
    version="1.0.0",
)


class WaterSample(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    pH: float = Field(..., title="pH — Acidity / Alkalinity")
    DO: float = Field(..., title="DO — Dissolved Oxygen")
    BOD: float = Field(..., title="BOD — Biochemical Oxygen Demand")
    Cond: float = Field(..., title="Cond — Electrical Conductivity")
    Nitrate: float = Field(..., title="Nitrate — Nitrate Concentration")
    COD: float = Field(..., title="COD — Chemical Oxygen Demand")
    TotalColi: float = Field(..., title="TotalColi — Total Coliform Count")
    FecalColi: float = Field(..., title="FecalColi — Faecal Coliform Count")
    Turbidity: float = Field(..., title="Turbidity — Water Cloudiness")
    TotAlk: float = Field(..., title="TotAlk — Total Alkalinity")
    Chloride: float = Field(..., title="Chloride — Chloride Concentration")
    Sulphate: float = Field(..., title="Sulphate — Sulphate Concentration")
    Hardness: float = Field(..., title="Hardness — Total Water Hardness")
    TDS: float = Field(..., title="TDS — Total Dissolved Solids")
    TSS: float = Field(..., title="TSS — Total Suspended Solids")
    Fluoride: float = Field(..., title="Fluoride — Fluoride Concentration")
    microbiological_verified: bool = Field(
        default=False,
        title="Microbiological Verification — Confirmed Laboratory Verification",
    )


# Load the exact saved objects used by the notebook.
classification_model = joblib.load(MODEL_DIR / "best_classification_model.pkl")
rf_imputer = joblib.load(PREP / "rf_imputer.pkl")
label_encoder = joblib.load(PREP / "label_encoder.pkl")
if_imputer = joblib.load(PREP / "if_imputer.pkl")
if_scaler = joblib.load(PREP / "if_scaler.pkl")
isolation_forest = joblib.load(ANOMALY_DIR / "isolation_forest_model.pkl")
lof_model = joblib.load(ANOMALY_DIR / "local_outlier_factor_model.pkl")
ocsvm_model = joblib.load(ANOMALY_DIR / "one_class_svm_model.pkl")

# Rebuild the training-reference statistics used for anomaly explanations.
raw_df = pd.read_csv(BASE / "nwmp_model_ready.csv")
train_metadata = pd.read_csv(PREP / "train_metadata.csv")

for frame in (raw_df, train_metadata):
    frame["station_id"] = pd.to_numeric(
        frame["station_id"], errors="raise"
    ).astype("Int64")
    frame["year"] = pd.to_numeric(
        frame["year"], errors="raise"
    ).astype("Int64")

raw_lookup = raw_df[["station_id", "year"] + FEATURES].copy()
train_raw = train_metadata.merge(
    raw_lookup,
    on=["station_id", "year"],
    how="left",
    validate="many_to_one",
)
train_reference = train_raw[FEATURES].apply(pd.to_numeric, errors="coerce")
train_median = train_reference.median()
train_iqr = (
    train_reference.quantile(0.75) - train_reference.quantile(0.25)
).replace(0, np.nan)

# Empirical training-domain bounds. Values outside these bounds are
# extrapolations and should be reviewed rather than trusted automatically.
train_min = train_reference.min()
train_max = train_reference.max()


def assess_drinking_suitability(measurements, microbiological_verified=False):
    failed_parameters = []
    warnings = []
    invalid_parameters = []
    missing_parameters = []

    for parameter, limits in BIS_LIMITS.items():
        value = measurements.get(parameter)

        if value is None:
            missing_parameters.append(parameter)
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            invalid_parameters.append(parameter)
            continue

        if not math.isfinite(value) or value < 0:
            invalid_parameters.append(parameter)
            continue

        if parameter == "pH":
            if value < limits["acceptable_min"] or value > limits["acceptable_max"]:
                failed_parameters.append({
                    "parameter": parameter,
                    "value": value,
                    "reason": "Outside pH range 6.5–8.5",
                })
        elif value > limits["permissible_max"]:
            failed_parameters.append({
                "parameter": parameter,
                "value": value,
                "permissible_limit": limits["permissible_max"],
                "unit": limits["unit"],
            })
        elif value > limits["acceptable_max"]:
            warnings.append({
                "parameter": parameter,
                "value": value,
                "acceptable_limit": limits["acceptable_max"],
                "permissible_limit": limits["permissible_max"],
                "unit": limits["unit"],
            })

    if failed_parameters:
        status = "NOT_SUITABLE"
        message = "One or more assessed parameters exceed the applicable permissible limit."
    elif invalid_parameters or missing_parameters:
        status = "INSUFFICIENT_DATA"
        message = "Some required drinking-water measurements are missing or invalid."
    elif not microbiological_verified:
        status = "INSUFFICIENT_DATA"
        message = (
            "Chemical screening completed, but microbiological laboratory "
            "verification is required."
        )
    elif warnings:
        status = "WITHIN_PERMISSIBLE_LIMITS"
        message = (
            "Some parameters exceed acceptable limits but remain within "
            "permissible limits."
        )
    else:
        status = "MEETS_ASSESSED_LIMITS"
        message = "All assessed chemical parameters meet the selected acceptable limits."

    return {
        "status": status,
        "message": message,
        "parameters_checked": (
            len(BIS_LIMITS) - len(missing_parameters) - len(invalid_parameters)
        ),
        "failed_parameters": failed_parameters,
        "warnings": warnings,
        "missing_parameters": missing_parameters,
        "invalid_parameters": invalid_parameters,
        "microbiological_verified": bool(microbiological_verified),
        "disclaimer": (
            "Preliminary chemical screening only. This does not certify "
            "drinking-water safety or complete BIS compliance."
        ),
    }


@app.get("/")
def home():
    return {
        "message": "Water Quality API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "api_status": "running",
        "classification_model": "loaded",
        "anomaly_models": "loaded",
        "mongodb_connected": False,
    }


@app.post("/predict")
def predict_water_quality(sample: WaterSample):
    try:
        sample_dict = sample.model_dump()
        microbiological_verified = sample_dict.pop("microbiological_verified")

        sample_df = pd.DataFrame(
            [[sample_dict[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

        # 1. Validate clearly invalid input before running ML models.
        # Zero is valid for some individual measurements (for example,
        # a verified zero coliform count), so only an all-zero sample is
        # rejected by this rule. Negative values and pH outside 0-14 are
        # also rejected as invalid measurements.
        invalid_input_reasons = []
        if all(float(sample_dict[feature]) == 0.0 for feature in FEATURES):
            invalid_input_reasons.append("ALL_MEASUREMENTS_ZERO")
        negative_inputs = [
            feature for feature in FEATURES if float(sample_dict[feature]) < 0
        ]
        if negative_inputs:
            invalid_input_reasons.append("NEGATIVE_MEASUREMENTS")
        if not 0.0 <= float(sample_dict["pH"]) <= 14.0:
            invalid_input_reasons.append("PH_OUTSIDE_PHYSICAL_SCALE")

        # Always run the preliminary chemical screen, but do not ask the
        # classifier to label a clearly invalid sample.
        drinking_result = assess_drinking_suitability(
            sample_dict, microbiological_verified
        )

        outside_training_range = []
        for feature in FEATURES:
            value = float(sample_dict[feature])
            lower = train_min[feature]
            upper = train_max[feature]
            if pd.notna(lower) and pd.notna(upper):
                if value < float(lower) or value > float(upper):
                    outside_training_range.append({
                        "parameter": feature,
                        "value": value,
                        "training_min": float(lower),
                        "training_max": float(upper),
                    })

        if invalid_input_reasons:
            invalid_messages = {
                "ALL_MEASUREMENTS_ZERO": (
                    "All 16 measurements are zero. This is not accepted as a "
                    "valid water sample; verify the laboratory readings."
                ),
                "NEGATIVE_MEASUREMENTS": (
                    "One or more measurements are negative; verify the input."
                ),
                "PH_OUTSIDE_PHYSICAL_SCALE": (
                    "pH must be within the conventional 0-14 scale."
                ),
            }
            return {
                "success": True,
                "water_quality": {
                    "prediction_status": "INVALID_INPUT",
                    "predicted_class": None,
                    "confidence": None,
                    "class_probabilities": {},
                    "review_required": True,
                    "review_reasons": invalid_input_reasons,
                    "message": (
                        "Classification was not performed because the input "
                        "failed basic validation. Correct the measurements and retry."
                    ),
                    "model_output_note": (
                        "No water-quality class was assigned. Invalid input must "
                        "not be interpreted as Excellent, Good, Poor, Unfit, or Very_poor."
                    ),
                },
                "drinking_suitability": drinking_result,
                "anomaly_detection": {
                    "status": "NOT_EVALUATED",
                    "consensus_votes": None,
                    "models": {
                        "isolation_forest": "NOT_EVALUATED",
                        "local_outlier_factor": "NOT_EVALUATED",
                        "one_class_svm": "NOT_EVALUATED",
                    },
                    "chemical_deviation_explanation": [],
                },
                "data_quality": {
                    "status": "INVALID_INPUT",
                    "warnings": [invalid_messages[key] for key in invalid_input_reasons],
                    "invalid_input_reasons": invalid_input_reasons,
                    "outside_training_range": outside_training_range,
                },
                "mongodb_saved": False,
            }

        # 2. Supervised water-quality classification.
        X_classification = rf_imputer.transform(sample_df)
        predicted_encoded = classification_model.predict(X_classification)[0]
        predicted_class = label_encoder.inverse_transform(
            np.array([int(predicted_encoded)])
        )[0]
        probabilities = classification_model.predict_proba(X_classification)[0]

        class_probabilities = {}
        for class_id, probability in zip(
            classification_model.classes_, probabilities
        ):
            class_name = label_encoder.inverse_transform(
                np.array([int(class_id)])
            )[0]
            class_probabilities[str(class_name)] = round(float(probability), 4)

        top_probability = float(max(probabilities))
        sorted_probabilities = sorted(probabilities, reverse=True)
        margin = (
            sorted_probabilities[0] - sorted_probabilities[1]
            if len(sorted_probabilities) > 1 else 1.0
        )

        # 3. Three-model anomaly consensus.
        X_anomaly = if_imputer.transform(sample_df)
        X_anomaly = if_scaler.transform(X_anomaly)

        if_flag = isolation_forest.predict(X_anomaly)[0] == -1
        lof_flag = lof_model.predict(X_anomaly)[0] == -1
        svm_flag = ocsvm_model.predict(X_anomaly)[0] == -1
        votes = sum([if_flag, lof_flag, svm_flag])

        if votes >= 2:
            anomaly_status = "ANOMALY"
        elif votes == 1:
            anomaly_status = "REVIEW_REQUIRED"
        else:
            anomaly_status = "NORMAL"

        # 4. Explain largest deviations from training median in IQR units.
        sample_series = pd.Series(sample_dict)[FEATURES]
        deviations = (
            (sample_series - train_median).abs() / train_iqr
        )
        deviations = pd.to_numeric(
            deviations, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        top_deviations = deviations.dropna().nlargest(5)

        explanation = [
            {
                "parameter": str(name),
                "deviation_score": round(float(value), 4),
            }
            for name, value in top_deviations.items()
        ]

        # 5. Input and training-domain checks.
        warnings = []

        negative = [feature for feature in FEATURES if sample_dict[feature] < 0]
        if negative:
            warnings.append("Negative measurements: " + ", ".join(negative))

        if sample_dict["BOD"] > sample_dict["COD"]:
            warnings.append("BOD is greater than COD; verify input.")

        # pH is expected to be within the conventional 0-14 scale.
        if not 0 <= sample_dict["pH"] <= 14:
            warnings.append("pH is outside the conventional 0-14 measurement scale.")

        # A value outside the training range is a model-domain limitation,
        # not automatically a data-quality error. Keep it visible separately
        # so legitimate individual zero measurements are not labelled invalid.
        model_domain_notes = []
        if outside_training_range:
            for item in outside_training_range:
                model_domain_notes.append(
                    f"{item['parameter']}={item['value']} is outside the observed "
                    f"training range ({item['training_min']} to {item['training_max']}). "
                    "This may be a valid laboratory measurement; interpret the model "
                    "prediction cautiously."
                )

        # The classifier output is retained for transparency, but an
        # out-of-domain, anomalous, or chemically unsuitable sample is flagged
        # for human review. This does not turn an ML class into a safety verdict.
        review_reasons = []
        if outside_training_range:
            review_reasons.append("OUTSIDE_TRAINING_RANGE")
        if anomaly_status == "ANOMALY":
            review_reasons.append("ANOMALY_DETECTED")
        if drinking_result["status"] == "NOT_SUITABLE":
            review_reasons.append("CHEMICAL_SCREENING_NOT_SUITABLE")
        if drinking_result["status"] == "INSUFFICIENT_DATA":
            review_reasons.append("DRINKING_SUITABILITY_INSUFFICIENT_DATA")
        if negative:
            review_reasons.append("NEGATIVE_MEASUREMENTS")
        if sample_dict["BOD"] > sample_dict["COD"]:
            review_reasons.append("BOD_GREATER_THAN_COD")

        confidence_review = bool(
            top_probability < 0.60 or margin < 0.15
        )
        if confidence_review:
            review_reasons.append("LOW_MODEL_CONFIDENCE_OR_MARGIN")

        review_required = bool(review_reasons)

        return {
            "success": True,
            "water_quality": {
                "predicted_class": str(predicted_class),
                "confidence": round(top_probability, 4),
                "class_probabilities": class_probabilities,
                "review_required": review_required,
                "review_reasons": review_reasons,
                "model_output_note": (
                    "The predicted class is the classifier output, not a drinking-water "
                    "safety certification. Review is required when input is outside the "
                    "training range, anomaly detection flags the sample, or chemical "
                    "screening is unsuitable or incomplete."
                ),
            },
            "drinking_suitability": drinking_result,
            "anomaly_detection": {
                "status": anomaly_status,
                "consensus_votes": int(votes),
                "models": {
                    "isolation_forest": "ANOMALY" if if_flag else "NORMAL",
                    "local_outlier_factor": "ANOMALY" if lof_flag else "NORMAL",
                    "one_class_svm": "ANOMALY" if svm_flag else "NORMAL",
                },
                "chemical_deviation_explanation": explanation,
            },
            "data_quality": {
                "status": "REQUIRES_REVIEW" if warnings else "PASS",
                "warnings": warnings,
                "outside_training_range": outside_training_range,
                "model_domain_status": (
                    "OUTSIDE_TRAINING_RANGE" if outside_training_range else "WITHIN_TRAINING_RANGE"
                ),
                "model_domain_note": (
                    "One or more values are outside the observed training-data range. "
                    "This is a model extrapolation warning, not proof that the laboratory "
                    "measurement is invalid." if outside_training_range else
                    "All values are within the observed training-data range."
                ),
                "model_domain_notes": model_domain_notes,
            },
            "mongodb_saved": False,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
