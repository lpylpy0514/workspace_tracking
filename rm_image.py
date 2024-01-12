import os


# 定义一个函数来递归删除每一层文件夹中名为"image"的文件夹内的文件
def delete_files_in_mask_folder(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for d in dirs:
            if d == "image" or d == "images" or d == "img":
                mask_folder = os.path.join(root, d)
                print(f"Deleting files in {mask_folder}")
                for f in os.listdir(mask_folder):
                    file_path = os.path.join(mask_folder, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"Files in {mask_folder} deleted")

        for d in dirs:
            next_folder = os.path.join(root, d)
            delete_files_in_mask_folder(next_folder)


# 调用函数，传入根文件夹路径
root_folder_path = "/media/ymz/T7/sam_mask"
delete_files_in_mask_folder(root_folder_path)