from roboflow import Roboflow


rf = Roboflow(api_key="Z6xAKEyYZ1KIz4ihwYCP")
project = rf.workspace("team-roboflow").project("coco-128")
version = project.version(2)
dataset = version.download("coco")
