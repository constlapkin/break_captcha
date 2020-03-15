from prepare import prepare_data as pd
from ai import train_model as tm, check_captcha as cc


def main():
    # Save and prepare data
    domain = "http://84.201.147.32"
    captcha_folder = "row_data"
    output_captcha_folder = "prepared_data"
    
    pd.parse(domain, captcha_folder)
    pd.cut_border(captcha_folder)
    pd.segmentation(captcha_folder, output_captcha_folder)
    
    # Train AI and check captcha
    output_captcha_folder = "prepared_data"
    model_filename = "models\\model_0001.hdf5"
    model_labels_filename = "labels\\label_0001.dat"
    tested_data = "test_data"
    tm.main(output_captcha_folder, model_filename, model_labels_filename)
    cc.main(model_filename, model_labels_filename, tested_data)


if __name__ == "__main__":
    main()
