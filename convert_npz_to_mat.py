import numpy as np
import scipy.io as sio
import os

def convert_npz_to_mat(npz_path, mat_path=None, target_max_volts=4.0):
    """
    ממיר קובץ npz לקובץ mat המותאם לאפליקציית המטלב.
    כולל הגברה בטוחה של האות למניעת צליל חלש.
    """
    # 1. טעינת קובץ ה-npz
    if not os.path.exists(npz_path):
        print(f"Error: The file {npz_path} does not exist.")
        return
        
    with np.load(npz_path) as data:
        # הדפסת השמות של המשתנים שנמצאים בתוך ה-npz כדי שתוכל לראות מה יש שם
        print("Variables found in NPZ:", list(data.keys()))
        
        # 2. חילוץ האות (משתנה פוטנציאלי - שנה את השם 'signal' במידת הצורך)
        # ננסה לנחש שמות נפוצים, או שניקח את המשתנה הראשון
        signal_key = 'audioSignal' if 'audioSignal' in data else ('signal' if 'signal' in data else list(data.keys())[0])
        raw_signal = data[signal_key]
        
        # 3. חילוץ קצב הדגימה (Fs)
        fs_key = 'Fs' if 'Fs' in data else ('fs' if 'fs' in data else ('sampling_rate' if 'sampling_rate' in data else None))
        if fs_key:
            fs = float(data[fs_key])
        else:
            fs = 500000.0  # ברירת מחדל של 500kHz אם לא נמצא בקובץ
            print(f"Warning: No Fs found in NPZ. Using default: {fs} Hz")

    # אילוץ וקטור עמודה חד-מימדי (עבור ערוץ בודד ao0 במטלב)
    raw_signal = raw_signal.flatten()

    # 4. נרמול והגברה דיגיטלית בטוחה (כדי שלא יישמע חלש!)
    # נגביר את האות כך שערך השיא שלו יהיה בדיוק target_max_volts (למשל 4V לצליל שמיע)
    max_val = np.max(np.abs(raw_signal))
    if max_val > 0:
        audio_signal_amplified = (raw_signal / max_val) * target_max_volts
    else:
        audio_signal_amplified = raw_signal

    # 5. שמירה בפורמט מטלב (.mat) עם השמות המדויקים שה-GUI מחפש
    if mat_path is None:
        mat_path = npz_path.replace('.npz', '.mat')
        
    mat_dict = {
        'audioSignal': audio_signal_amplified,
        'Fs': fs
    }
    
    sio.savemat(mat_path, mat_dict)
    
    print("-" * 40)
    print(f"Successfully converted!")
    print(f"Saved to: {mat_path}")
    print(f"Signal shape: {audio_signal_amplified.shape}")
    print(f"Max Peak Voltage: {np.max(np.abs(audio_signal_amplified))} V")
    print(f"Sample Rate (Fs): {fs} Hz")
    print("-" * 40)

# --- דוגמה לשימוש בסקריפט ---
if __name__ == "__main__":
    # נתיב לקובץ ה-npz שלך
    input_npz = "C:\\Users\\noam4\\OneDrive\\Desktop\\7.npz"
    
    # הגדרת מתח המטרה (4.0V לצליל השמיע שאתה בודק עכשיו. ל-USV האמיתי תעלה ל-7.0 או 8.0)
    convert_npz_to_mat(input_npz, target_max_volts=7.0)
