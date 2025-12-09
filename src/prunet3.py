import torch
from safetensors.torch import load_file, save_file
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Trỏ đến file t3_cfg.safetensors trong thư mục checkpoint của bạn
checkpoint_path = "/content/drive/MyDrive/checkpoints/chatterbox_vietnamese_multispeaker_v3/checkpoint-3000/model.safetensors"
final_path = "/content/drive/MyDrive/checkpoints/chatterbox_vietnamese_multispeaker_v3/checkpoint-3000/t3_cfg.safetensors"

# --------------------------
output_path = "/content/chatterbox-finetuning/infer/t3_cfg.safetensors"
print(f"Đang xử lý file: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print("❌ Lỗi: Không tìm thấy file checkpoint! Kiểm tra lại đường dẫn.")
else:
    # 1. Load file checkpoint hiện tại
    try:
        state_dict = load_file(checkpoint_path)
    except:
        # Phòng trường hợp nó là file .pt/.bin chứ không phải safetensors
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 2. Tạo dict mới và sửa tên key
    new_state_dict = {}
    fixed_count = 0
    for key, value in state_dict.items():
        if key.startswith("t3."):
            # Cắt bỏ 3 ký tự đầu ("t3.")
            new_key = key[3:]
            new_state_dict[new_key] = value
            fixed_count += 1
        else:
            new_state_dict[key] = value

    # 3. Lưu đè lại file cũ (hoặc lưu ra file mới tùy bạn)
    if fixed_count > 0:
        print(f"✅ Đã tìm thấy và sửa {fixed_count} keys bị thừa 't3.'.")
        # Lưu file mới đã sửa
        save_file(new_state_dict, output_path)
        print(f"🎉 Đã lưu file đã sửa tại: {output_path}")
        save_file(new_state_dict, final_path)
        print("👉 Bây giờ bạn có thể chạy lại code Gradio!")
    else:
        print("⚠️ File này có vẻ đã sạch (không có prefix 't3.'), không cần sửa.")
