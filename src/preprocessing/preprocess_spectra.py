import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


# Параметры сглаживания
SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 1
SAVGOL_DERIV = 0

# Параметры удаления фона
POLY_DEGREE = 5


def load_task(task_dir: str):
    """
    :param task_dir: путь к папке задачи
    :return: кортеж (metadata, spectra)
    """
    metadata = pd.read_csv(os.path.join(task_dir, 'metadata.csv'))
    spectra = pd.read_csv(os.path.join(task_dir, 'spectra.csv'))
    return metadata, spectra


def get_wavelengths(spectra: pd.DataFrame):
    """
    :param spectra: матрица спектров с заголовками-длинами волн
    :return: numpy-массив длин волн
    """
    wavelengths = []
    for col in spectra.columns:
        try:
            wavelengths.append(float(str(col).replace(',', '.')))
        except ValueError:
            wavelengths.append(np.nan)
    return np.array(wavelengths)


def crop_range(spectra: pd.DataFrame, wavelengths: np.ndarray, lo: float, hi: float):
    """
    :param spectra: матрица спектров
    :param wavelengths: массив длин волн
    :param lo: нижняя граница диапазона
    :param hi: верхняя граница диапазона
    :return: кортеж (cropped_spectra, cropped_wavelengths)
    """
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    idx = np.where(mask)[0]
    cropped = spectra.iloc[:, idx].copy()
    return cropped, wavelengths[idx]


def apply_savgol(spectra: pd.DataFrame):
    """
    :param spectra: матрица спектров
    :return: сглаженная матрица спектров
    """
    data = spectra.values.astype(np.float64)
    smoothed = savgol_filter(
        data,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
        deriv=SAVGOL_DERIV,
        axis=1
    )
    return pd.DataFrame(smoothed, columns=spectra.columns)


def remove_baseline_poly(spectra: pd.DataFrame, wavelengths: np.ndarray):
    """
    :param spectra: матрица спектров
    :param wavelengths: массив длин волн
    :return: матрица спектров без фона
    """
    data = spectra.values.astype(np.float64)
    x = wavelengths.astype(np.float64)
    result = np.zeros_like(data)

    for i in range(data.shape[0]):
        y = data[i]
        coeffs = np.polyfit(x, y, POLY_DEGREE)
        baseline = np.polyval(coeffs, x)
        result[i] = y - baseline

    return pd.DataFrame(result, columns=spectra.columns)


def normalize_std(spectra: pd.DataFrame):
    """
    :param spectra: матрица спектров
    :return: нормированная матрица спектров
    """
    data = spectra.values.astype(np.float64)
    stds = data.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    normalized = data / stds
    return pd.DataFrame(normalized, columns=spectra.columns)


def process_scenario1(spectra: pd.DataFrame, wavelengths: np.ndarray):
    """
    :param spectra: матрица спектров
    :param wavelengths: массив длин волн
    :return: обработанная матрица спектров
    """
    cropped, _ = crop_range(spectra, wavelengths, lo=-np.inf, hi=1900)
    smoothed = apply_savgol(cropped)
    normalized = normalize_std(smoothed)
    return normalized


def process_scenario2(spectra: pd.DataFrame, wavelengths: np.ndarray):
    """
    :param spectra: матрица спектров
    :param wavelengths: массив длин волн
    :return: обработанная матрица спектров
    """
    cropped, wl_cropped = crop_range(spectra, wavelengths, lo=1000, hi=1900)
    smoothed = apply_savgol(cropped)
    corrected = remove_baseline_poly(smoothed, wl_cropped)
    normalized = normalize_std(corrected)
    return normalized


def save_scenario(metadata: pd.DataFrame, spectra: pd.DataFrame, out_dir: str):
    """
    :param metadata: таблица с метаданными
    :param spectra: матрица спектров
    :param out_dir: папка для сохранения
    :return: None
    """
    os.makedirs(out_dir, exist_ok=True)
    metadata.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
    spectra.to_csv(os.path.join(out_dir, 'spectra.csv'), index=False)


def main():
    """
    :return: None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    processed_root = os.path.join(project_root, 'data', 'processed')
    tasks = ['task_malignant_vs_benign', 'task_melanoma_vs_nevus_sk']

    for task in tasks:
        task_dir = os.path.join(processed_root, task)
        if not os.path.isdir(task_dir):
            print(f'Пропущена задача: {task_dir}')
            continue

        print('=' * 60)
        print(f'Обработка задачи: {task}')
        print('=' * 60)

        metadata, spectra = load_task(task_dir)
        wavelengths = get_wavelengths(spectra)

        print(f'Образцов: {len(metadata)}, точек: {spectra.shape[1]}')
        print(f'Диапазон: {np.nanmin(wavelengths):.1f} – {np.nanmax(wavelengths):.1f} см⁻¹')

        # Сценарий 1
        s1 = process_scenario1(spectra, wavelengths)
        out1 = os.path.join(task_dir, 'scenario1')
        save_scenario(metadata, s1, out1)
        print(f'Сценарий 1 сохранён: {out1} ({s1.shape[1]} точек)')

        # Сценарий 2
        s2 = process_scenario2(spectra, wavelengths)
        out2 = os.path.join(task_dir, 'scenario2')
        save_scenario(metadata, s2, out2)
        print(f'Сценарий 2 сохранён: {out2} ({s2.shape[1]} точек)')

        print()


if __name__ == '__main__':
    main()