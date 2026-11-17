from typing import Dict, Literal

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    reading_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Note de lecture comprise entre 0 et 100",
    )

    writing_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Note d'écriture comprise entre 0 et 100",
    )

    parental_level_of_education: Literal[
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ]

    test_preparation_course: Literal[
        "none",
        "completed",
    ]

    gender: Literal[
        "female",
        "male",
    ]

    race_ethnicity: Literal[
        "group A",
        "group B",
        "group C",
        "group D",
        "group E",
    ]

    lunch: Literal[
        "standard",
        "free/reduced",
    ]


class PredictionResponse(BaseModel):
    class_name: str
    probabilities: Dict[str, float]