import albumentations as A
import librosa
from audiomentations import Compose, TimeShift, PitchShift, TimeStretch, AddGaussianNoise, Gain
import numpy as np
import os
import random
from glob import glob
from PIL import Image


def get_augmented_image_by_mode(image_path_or_pil, mode: int):
    """
    선택한 모드에 따라 지정된 데이터 증강을 적용
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")
    image_np = np.array(image)

    if mode == 1:
        # 좌우 반전
        transform = A.Compose([A.HorizontalFlip(p=1.0)])
    elif mode == 2:
        # 무작위 회전 (-30도 ~ 30도)
        transform = A.Compose([A.Rotate(limit=30, border_mode=0, value=(255, 255, 255), p=1.0)])
    elif mode == 3:
        # 색상 미묘하게 변경
        transform = A.Compose([A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0)])
    elif mode == 4:
        # 노이즈 추가
        transform = A.Compose([A.GaussNoise(var_limit=(20.0, 50.0), p=1.0)])
    elif mode == 5:
        # 해상도 낮춤
        transform = A.Compose([A.Downscale(scale_min=0.4, scale_max=0.6, p=1.0)])
    elif mode == 6:
        # 위치 이동 및 크기 조절 (포켓몬이 사방으로 움직이고 크기가 변함)
        transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, border_mode=0,
                                                  value=(255, 255, 255), p=1.0)])
    elif mode == 7:
        # 컷아웃 (이미지 일부를 무작위로 가려서 힌트를 숨김)
        # 픽셀 크기에 맞게 hole의 크기를 적절히 조절해야함 (예: 16x16 ~ 32x32 크기 구멍 3개)
        transform = A.Compose([A.CoarseDropout(max_holes=3, max_height=32, max_width=32, fill_value=0, p=1.0)])
    elif mode == 8:
        # 선명도 강화 (외곽선 테두리를 뚜렷하게 만듦)
        transform = A.Compose([A.Sharpen(alpha=(0.3, 0.5), lightness=(0.5, 1.0), p=1.0)])
    else:
        # 잘못된 번호가 들어오면 변형 없이 원본 반환
        return Image.fromarray(image_np)

    # 3. 증강 적용 및 반환
    augmented = transform(image=image_np)
    return Image.fromarray(augmented['image'])

def balance_and_augment_dataset(src_train_dir, tgt_train_dir, target_count=60):
    """
    src_train_dir: 원본 train 폴더 경로
    tgt_train_dir: 증강된 이미지가 저장될 새 폴더 경로
    target_count: 클래스당 맞춰줄 목표 이미지 장수
    """
    classes = os.listdir(src_train_dir)

    for cls in classes:
        cls_path = os.path.join(src_train_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        out_cls_path = os.path.join(tgt_train_dir, cls)
        os.makedirs(out_cls_path, exist_ok=True)
        img_paths = glob(os.path.join(cls_path, "*.png"))
        img_count = len(img_paths)

        if img_count == 0:
            continue

        print(f"[{cls}] 원본 데이터 {img_count}장 ➡️ {target_count}장으로 증강 중...")

        for idx, p in enumerate(img_paths):
            img = Image.open(p)
            img.save(os.path.join(out_cls_path, f"origin_{idx}.png"))

        for i in range(target_count - img_count):
            random_img_path = random.choice(img_paths)
            random_mode = random.randint(1, 8)
            aug_img = get_augmented_image_by_mode(random_img_path, mode=random_mode)
            aug_img.save(os.path.join(out_cls_path, f"aug_{i}_mode{random_mode}.png"))
# 사용 예시
#balance_and_augment_dataset("C:\Users\zocla\Desktop\강의\4-1_딥러닝응용\기말프젝\gen1-pokemon-images\dataset\train", "C:\Users\zocla\Desktop\강의\4-1_딥러닝응용\기말프젝\gen1-pokemon-images\dataset\train", target_count=60)

def get_augmented_audio_by_mode(audio_path, mode: int):
    """
    .ogg 오디오 파일 경로를 받아 지정된 모드별 증강을 적용하고,
    오디오 데이터(numpy)와 샘플 레이트(sr)를 반환하는 함수
    """
    y, sr = librosa.load(audio_path, sr=None)
    if mode == 1:
        # [시간 이동] 최대 0.2초 내외로 소리를 앞뒤로 미룸
        transform = Compose([TimeShift(min_fraction=-0.2, max_fraction=0.2, p=1.0)])
    elif mode == 2:
        # [피치 조절] 음 높낮이를 -3~3 반음(semitones) 범위 내에서 조절
        transform = Compose([PitchShift(min_semitones=-3, max_semitones=3, p=1.0)])
    elif mode == 3:
        # [속도 조절] 0.8배속에서 1.2배속 사이로 조절
        transform = Compose([TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0)])
    elif mode == 4:
        # [노이즈 추가] 미세한 가우시안 노이즈 추가
        transform = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0)])
    elif mode == 5:
        # [볼륨 조절] 소리 크기를 -6dB ~ +6dB 범위 내에서 변경
        transform = Compose([Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0)])
    else:
        # 잘못된 번호는 변형 없이 원본 반환
        return y, sr
    augmented_y = transform(samples=y, sample_rate=sr)

    return augmented_y, sr