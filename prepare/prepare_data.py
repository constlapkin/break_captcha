import requests
from bs4 import BeautifulSoup
import os.path
from PIL import Image
import cv2
import glob


def parse(domain="http://84.201.147.32", captcha_folder="row_data"):
    """
    Parse from 'domain' (php file from CMS Bitrix, where will generate n-captches)
    and saved all images to 'captcha_folder' with name from <span> tag.

    :param domain: url to Bitrix site, like http://84.201.147.32
    :param captcha_folder: dir for save images
    
    Below it's file from Bitrix that generate n-captchas.
    /bitrix/tools/captcha_test.php:

        <?
        define("NO_KEEP_STATISTIC", "Y");
        define("NO_AGENT_STATISTIC","Y");
        define("NOT_CHECK_PERMISSIONS", true);
        $HTTP_ACCEPT_ENCODING = "";
        $_SERVER["HTTP_ACCEPT_ENCODING"] = "";
        require($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/prolog_before.php");
        ?>
        <?for($i = 0; $i < 90; $i++):?>
        <?$arResult["CAPTCHA_CODE"] = $APPLICATION->CaptchaGetCode();?>
        <div style="display: inline-block">
            <img src="/bitrix/tools/captcha.php?captcha_sid=<?echo $arResult["CAPTCHA_CODE"]?>"
                width="180" height="40" alt="CAPTCHA" /><br>
            <?
            global $DB;
            $res = $DB->Query("SELECT CODE FROM b_captcha WHERE ID = '".$DB->ForSQL($arResult["CAPTCHA_CODE"],32)."' ");
            if($ar = $res->Fetch()):?>
                <span><?=$ar['CODE'];?></span>
            <?endif;?>
        </div>
        <?endfor;?>
        <?require($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/epilog_after.php");?>
    """
    
    url = domain + "/bitrix/tools/captcha_test.php"
    path = "\\" + captcha_folder + "\\"
    r = requests.get(url)
    content = r.text.encode('utf-8')

    soup = BeautifulSoup(content, "html.parser")
    captcha = soup.findAll('div')

    i = 1
    for el in captcha:
        print(i)
        i += 1
        code = el.find('span').text
        r1 = requests.get(domain + el.find('img').get('src'))
        with open(path + '{}.jpg'.format(code), 'wb') as img_file:
            img_file.write(r1.content)

    return 1


def cut_border(captcha_folder="row_data"):
    """
    It takes files from 'captcha_folder' and cut bitrix border on captcha images (2px by all sides)
    :param captcha_folder: dir that has captcha images
    """
    captcha_image_files = glob.glob(os.path.join(captcha_folder, "*"))
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("Processing image {}/{}".format(i + 1, len(captcha_image_files)))
        img = Image.open(captcha_image_file)
        width = img.size[0]
        height = img.size[1]
        img3 = img.crop((2, 2, width - 2, height - 2))
        img3.save(captcha_image_file)
    return 1


def segmentation(captcha_folder="row_data", output_captcha_folder="prepared_data"):
    """
    It takes images from 'captcha_folder', divides and saves to 'output_captcha_folder' by letter from name picture.
    Segmentation on CV2 library: 
    cvtColor - cv2.COLOR_BGR2GRAY
    threshold - cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    getStructuringElement - cv2.MORPH_CROSS, (1, 3)
    erode
    morphologyEx - cv2.MORPH_OPEN
    findContours - cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    
    :param captcha_folder: dir that has captcha images
    :param output_captcha_folder: dir that will be has prepared captcha images
    
    Result by image will be only if segmentation finds 5 elements.
    """
    captcha_image_files = glob.glob(os.path.join(captcha_folder, "*"))
    counts = {}
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("Processing image {}/{}".format(i + 1, len(captcha_image_files)))
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]
        image = cv2.imread(captcha_image_file)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, None, 255)

        res, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # cv2.imshow("contours", gray)
        # cv2.waitKey(0)

        letter_image_regions = []

        for contour in contours:

            (x, y, w, h) = cv2.boundingRect(contour)

            if w < 8 or h < 8:
                continue

            if w / h > 0.8 and w < 30:
                half_width = w / 2
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))

            else:
                letter_image_regions.append((x, y, w, h))

        if len(letter_image_regions) != 5:
            continue

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
            x, y, w, h = letter_bounding_box
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

            save_path = os.path.join(output_captcha_folder, letter_text)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.jpg".format(str(count).zfill(6)))

            cv2.imwrite(p, letter_image)

            counts[letter_text] = count + 1
    return 1


if __name__ == '__main__':
    parse()
    cut_border()
    segmentation()

