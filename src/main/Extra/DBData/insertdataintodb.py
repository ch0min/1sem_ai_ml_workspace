import pandas as pd
from sqlalchemy import create_engine

# Database connection details (replace with your details)
username = "sa"
password = "Europa123"
server = "localhost"
database = "DB_Soft"

# Create the engine using pyodbc
connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string)

# Load the combined CSV file
csv_file = "./combined_copy.csv"
df = pd.read_csv(csv_file)

# Define the mapping from CSV columns to database tables
tables_columns_mapping = {
    "Country": ["CountryName", "Region"],
    "City": ["Organisation", "CityName", "C40", "CityLocation"],
    "CityDetails": ["LandArea", "AverageAltitude"],
    "EmissionsReductionTargets": [
        "ReportingYear",
        "TargetBoundary",
        "BaselineYear",
        "BaselineEmissions",
        "PercentageReductionTarget",
        "TargetData",
        "Comment",
        "TypeOfTarget",
        "Scope",
        "EstimatedBAUAbsoluteEmissions",
        "IntensityUnit",
    ],
    "CitywideEmissions": [
        "ReportingYear",
        "MeasurementYear",
        "Boundary",
        "GasesIncluded",
        "TotalCityWideEmissions",
        "IncreaseDecreaseLastYear",
        "ReasonForIncreaseDecrease",
    ],
    "ClimateRiskAndVulnerabilityAssessments": [
        "Questionnaire",
        "AssessmentAttachmentOrLink",
        "BoundaryAssessment",
        "YearOfPublicationOrApproval",
        "FactorsConsidered",
        "PrimaryAuthors",
        "AdaptationGoalOrPlan",
        "LastUpdate",
    ],
    "Methodology": ["PrimaryMethodology", "MethodologyDetails"],
    "PopulationData": [
        "CurrentPopulation",
        "PopulationYear",
        "CityGDP",
        "GDPYear",
        "GDPSource",
        "GDPCurrency",
        "AverageAnnualTemperature",
        "CurrentPopulationYear",
    ],
}


# Insert data into each table
for table, columns in tables_columns_mapping.items():
    table_df = df[columns].dropna()  # Modify as needed
    table_df.to_sql(table, engine, if_exists="append", index=False)
    print(f"Data inserted into {table}")

print("Data insertion complete.")
