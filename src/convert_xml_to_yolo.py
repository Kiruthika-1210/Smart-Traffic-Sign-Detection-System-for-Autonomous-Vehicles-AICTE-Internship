import os
import xml.etree.ElementTree as ET

xml_folder = "../data/annotations"
output_folder = "../data/labels"
os.makedirs(output_folder, exist_ok=True)

# Automatically detect all classes from XML
detected_classes = set()

# First pass: read all XMLs and collect class names
for file in os.listdir(xml_folder):
    if file.endswith(".xml"):
        xml_path = os.path.join(xml_folder, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            cls = obj.find("name").text.strip()
            detected_classes.add(cls)

# Convert set to sorted list
classes = sorted(list(detected_classes))
print("✅ Detected classes:", classes)

def convert_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    yolo_data = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        cls_id = classes.index(cls_name)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_data.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    return yolo_data


# Second pass: convert XML to YOLO
for file in os.listdir(xml_folder):
    if file.endswith(".xml"):
        xml_path = os.path.join(xml_folder, file)
        yolo_output = convert_to_yolo(xml_path)

        txt_name = file.replace(".xml", ".txt")
        out_path = os.path.join(output_folder, txt_name)

        with open(out_path, "w") as f:
            for line in yolo_output:
                f.write(line + "\n")

print("✅ XML → YOLO conversion completed successfully!")
print("✅ Final class list:", classes)
