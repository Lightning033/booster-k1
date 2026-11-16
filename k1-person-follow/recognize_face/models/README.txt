Put the two OpenCV ONNX models here:

  face_detection_yunet_2023mar.onnx     (YuNet, finds faces)
  face_recognition_sface_2021dec.onnx   (SFace, turns a face into an embedding)

Easiest source is the earlier project on this workstation:

  cp ../booster-k1-person-recognition/recognize_face/models/*.onnx .

Otherwise they are in OpenCV's model zoo (opencv/opencv_zoo on GitHub), under
models/face_detection_yunet/ and models/face_recognition_sface/.

Check they loaded:

  python recognize_face/recognize_face.py

These files are in .gitignore because they are binaries, not because they are
sensitive. The folder that IS sensitive is face_db/people/.
