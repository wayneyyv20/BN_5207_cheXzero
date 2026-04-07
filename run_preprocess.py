import argparse
from pathlib import Path
from data_process import get_cxr_paths_list, img_to_hdf5, get_cxr_path_csv, write_report_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_out_path', type=str, default='data/cxr_paths.csv', help="Directory to save paths to all chest x-ray images in dataset.")
    parser.add_argument('--cxr_out_path', type=str, default='data/cxr.h5', help="Directory to save processed chest x-ray image data.")
    parser.add_argument('--dataset_type', type=str, default='mimic', choices=['mimic', 'chexpert-test','custom'], help="Type of dataset to pre-process")
    parser.add_argument('--mimic_impressions_path', default='data/mimic_impressions.csv', help="Directory to save extracted impressions from radiology reports.")
    parser.add_argument('--chest_x_ray_path', default='/deep/group/data/mimic-cxr/mimic-cxr-jpg/2.0.0/files', help="Directory where chest x-ray image data is stored. This should point to the files folder from the MIMIC chest x-ray dataset.")
    parser.add_argument('--radiology_reports_path', default='/deep/group/data/med-data/files/', help="Directory radiology reports are stored. This should point to the files folder from the MIMIC radiology reports dataset.")
    parser.add_argument('--custom_csv', type=str, default='train.csv', help="Path to custom CSV file (for dataset_type='custom')")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.dataset_type == "mimic":
        # Write Chest X-ray Image HDF5 File
        get_cxr_path_csv(args.csv_out_path, args.chest_x_ray_path)
        cxr_paths = get_cxr_paths_list(args.csv_out_path)
        img_to_hdf5(cxr_paths, args.cxr_out_path)

        #Write CSV File Containing Impressions for each Chest X-ray
        write_report_csv(cxr_paths, args.radiology_reports_path, args.mimic_impressions_path)
    elif args.dataset_type == "chexpert-test": 
        # Get all test paths based on cxr dir
        cxr_dir = Path(args.chest_x_ray_path)
        cxr_paths = list(cxr_dir.rglob("*.jpg"))
        cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths)) # filter only first frontal views 
        cxr_paths = sorted(cxr_paths) # sort to align with groundtruth
        assert(len(cxr_paths) == 500)
       
        img_to_hdf5(cxr_paths, args.cxr_out_path)
    
    elif args.dataset_type == "custom":
        import pandas as pd
        import os

        # 读取用户指定的 CSV
        df = pd.read_csv(args.custom_csv)

        # 检查必须的列
        required_cols = ['image_path', 'impression']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV 文件必须包含列: {required_cols}，实际列: {df.columns.tolist()}")

        # 获取图像路径列表（假设已经是绝对路径，若为相对路径请自行拼接 base_dir）
        cxr_paths = df['image_path'].tolist()

        # 可选：检查路径是否存在，过滤无效路径
        valid_mask = [os.path.exists(p) for p in cxr_paths]
        if not all(valid_mask):
            missing = [p for p, m in zip(cxr_paths, valid_mask) if not m]
            print(f"警告：以下 {len(missing)} 个图像路径不存在，将跳过。示例: {missing[:3]}")
            df = df[valid_mask].reset_index(drop=True)
            cxr_paths = df['image_path'].tolist()

        # 生成 HDF5 文件（图像顺序与 CSV 行顺序一致）
        print(f"正在将 {len(cxr_paths)} 张图像写入 {args.cxr_out_path} ...")
        img_to_hdf5(cxr_paths, args.cxr_out_path)

        # 保存 impressions 文本（用于 Zero-shot 提示）
        # 使用完整的 'impression' 列，每行一个文本
        impressions = df['impression'].tolist()
        pd.DataFrame(impressions, columns=['impression']).to_csv(args.mimic_impressions_path, index=False)
        print(f"Impressions 文本已保存至 {args.mimic_impressions_path}")

        # （可选）保存完整信息备份，包含 label_name 和 label，便于后续分析
        if 'label_name' in df.columns and 'label' in df.columns:
            info_csv = args.mimic_impressions_path.replace('.csv', '_full_info.csv')
            df[['image_path', 'impression', 'label_name', 'label']].to_csv(info_csv, index=False)
            print(f"完整信息已保存至 {info_csv}")
        
    


