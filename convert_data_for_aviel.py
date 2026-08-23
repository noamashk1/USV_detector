import pandas as pd

#txt_path = "Z:\\Shared\\Noam\\results\\asd_juv_06_05_2026\\filtered_results.txt"
#txt_path = "Z:\\Shared\\Noam\\results\\asd_juv_01_06_2026\\asd_juv_01_06_2026.txt"
#txt_path = "Z:\\Shared\\Noam\\results\\asd_juv_30_06_2026\\asd_juv_30_06_2026.txt"
txt_path = "Z:\\Shared\\Noam\\results\\asd_juv_04_08_2026\\asd_juv_04_08_2026.txt"
df = pd.read_csv(txt_path)
#print(list(df.columns))
df = df.rename(columns={
    "mouse ID": "mouse_name",
    "level": "levelname",
    "stim name": "freq_played",
    "stim index": "stimID",
    "start time": "time",
    "licks_time": "Lick",
})

#print(list(df.columns))
df = df.drop(columns=["end time", "go\\no-go", "licks_time_RD"])
# Print unique mouse names before filtering
print("Unique mouse names before filter:", df["mouse_name"].unique())



if txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_06_05_2026\\filtered_results.txt":
    df = df[df["mouse_name"].isin(['0008301EF1', '00082F8507', '0008301F0D'])]
elif txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_01_06_2026\\asd_juv_01_06_2026.txt":
    df = df[df["mouse_name"].isin(['000830272F', '0008302249', '00083027BD', '00082FD23A'])]
elif txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_30_06_2026\\asd_juv__30_2026.txt":
    df = df[df["mouse_name"].isin(['000830172E', '00082F78D7', '00082F8C8F', '000830265C', '0008300D79'])]
elif txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_04_08_2026\\asd_juv_04_08_2026.txt":
    df = df[df["mouse_name"].isin(['0008302650', '0008302118','00083020B9','0008302364'])]

if txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_30_06_2026\\asd_juv__30_2026.txt":
    # סנן את הנתונים כך שישארו רק עד התאריך 16/7/2026 כולל (פורמט 2026-07-16 בעמודה הראשונה)
    df = df[df.iloc[:, 0] <= "2026-07-16"]
    print("Unique dates after filtering:", df.iloc[:, 0].unique())

# Print unique mouse names after filtering
print("Unique mouse names after filter:", df["mouse_name"].unique())

# allowed_mouse_names = [
#     "000830272F",
#     "0008302249",
#     "00083027BD",
#     "00082FD23A"
# ]
# # 0008301EF1
# # 00082F8507
# # 0008301F0D
# df = df[df["mouse_name"].isin(allowed_mouse_names)]
print()  # תדפסיס רווח- כלומר ירידת שורה

print("unique values in 'score' column:", df["score"].unique())
df = df[df["score"].isin(['hit', 'miss', 'fa', 'cr','catch - no response' ,'catch - response'])] #תמחק את הטריילים של RD (שהיו בטעות)
print("unique values in 'score' column:", df["score"].unique())
score_mapping = {
    'hit': 0,
    'fa': 1,
    'miss': 2,
    'cr': 3,
    'catch - no response': 5,
    'catch - response': 6
}
df['score'] = df['score'].map(score_mapping)

desired_cols = [
    "mouse_num",
    "mouse_name",
    "level",
    "score",
    "freq_played",
    "trialNum",
    "stimID",
    "time",
    "levelname",
    "IR",
    "Lick",
    "level_day",
    "table_num",
    "age",
    "sex",
    "type"
]



for col in desired_cols:
    if col not in df.columns:
        df[col] = "-"
if txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_01_06_2026\\asd_juv_01_06_2026.txt":
    df["age"] = 25
    df["sex"] = "male"
elif txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_06_05_2026\\filtered_results.txt":
    df["age"] = 24
    df["sex"] = "female"
elif txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_30_06_2026\\asd_juv__30_2026.txt":
    df["age"] = 27
elif txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_04_08_2026\\asd_juv_04_08_2026.txt":
    df["age"] = 27
    # קובעים רשימות מפורשות של זכרים ונקבות
    # male_mice = ["000830172E", "00082F78D7"]
    # female_mice = ["00082F8C8F", "000830265C", "0008300D79"]
    male_mice = ["0008302118","00083020B9"]
    female_mice = ["0008302650","0008302364"]
    df["sex"] = df["mouse_name"].apply(lambda name: "male" if name in male_mice else ("female" if name in female_mice else "-"))
    undefined_sex_mice = df[df["sex"] == "-"]["mouse_name"].unique()
    if len(undefined_sex_mice) > 0:
        print("שים לב: העכברים הבאים סומנו עם '-' בעמודת sex:", ", ".join(undefined_sex_mice))

if txt_path == "Z:\\Shared\\Noam\\results\\asd_juv_30_06_2026\\asd_juv__30_2026.txt":
    df["type"] = "control"
else:
    df["type"] = "asd"


df["freq_played"] = (
    df["freq_played"]
    .astype(str)
    .str.replace("-", ".", regex=False)
    .str.replace(".npz", "", regex=False)
)


# סידור העמודות לפי הסדר הרצוי
df = df[desired_cols]
# יוצרים מילון שממפה כל שם עכבר (unique) למספר רץ שמתחיל ב-1
mouse_names = df["mouse_name"].unique()
mouse_name_to_num = {name: idx+1 for idx, name in enumerate(mouse_names)}

# ממלאים את העמודה mouse_num לפי המיפוי
df["mouse_num"] = df["mouse_name"].map(mouse_name_to_num)

print("Number of rows in the dataframe:", len(df))

# הדפסת כל השלישיות הייחודיות של stimID, freq_played ו-levelname שמופיעים באותה שורה
# unique_triplets = df[["stimID", "freq_played", "levelname"]].drop_duplicates()
# print("שלישיות ייחודיות של stimID, freq_played ו-levelname שמופיעות באותה שורה:")
# for _, row in unique_triplets.iterrows():
#     print(f"stimID: {row['stimID']}, freq_played: {row['freq_played']}, levelname: {row['levelname']}")


# for mouse_name in df['mouse_name'].unique():
#     mouse_df = df[df['mouse_name'] == mouse_name]
#     for level in mouse_df['levelname'].unique():
#         count = mouse_df[mouse_df['levelname'] == level].shape[0]
#         print(f"Mouse: {mouse_name}, Level: {level}, Number of rows: {count}")

# for mouse_name in df['mouse_name'].unique():
#     mouse_df = df[df['mouse_name'] == mouse_name]
#     l2_df = mouse_df[mouse_df['levelname'] == 'L2_with_catch']
#     total_l2_trials = l2_df.shape[0]
#     print(f"\nMouse: {mouse_name} - L2 frequency distribution:")
#     if total_l2_trials == 0:
#         print("  No L2 trials found for this mouse.")
#         continue
#     freq_counts = l2_df['freq_played'].value_counts(normalize=True) * 100
#     for freq, percent in freq_counts.items():
#         print(f"  Frequency: {freq}, Percent: {percent:.1f}%")


#print(list(df.columns))
import scipy.io as sio
import os

# שמור את ה-DataFrame כקובץ .mat תמיד בתיקיה C:\noam_projects\USV_detector, עם שם קובץ המקור + "_filterd" בסוף, במקום הסיומת.
save_dir = r"C:\\noam_projects\\USV_detector"
# Extract the parent directory name of the text file (not the file name itself)
parent_dir_name = os.path.basename(os.path.dirname(txt_path))
mat_file_name = parent_dir_name + "_filterd.mat"
mat_path = os.path.join(save_dir, mat_file_name)
print(mat_path)

sio.savemat(mat_path, {"data": df.to_records(index=False)})

print(f"Saved dataframe to {mat_path}")