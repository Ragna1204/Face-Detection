import cv2
import torchvision.transforms as transforms
import torch
import MER

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 30)

device = MER.get_default_device()
print(f"Using device: {device}")

w = './MERcnn.pth'
model = MER.to_device(MER.MERcnn(), device)

if str(device) == 'cpu':
    model.load_state_dict(torch.load(w, map_location=torch.device('cpu')), strict=False)
    print("Model loaded on CPU.")
if str(device) == 'gpu':
    model.load_state_dict(torch.load(w, map_location=torch.device('cuda')))
    print("Model loaded on GPU.")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

while True:
    _, frame = cam.read()
    if _:
        Bbox = MER.face_box(frame)
        if len(Bbox) > 0:
            for box in Bbox:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                faceExp = frame[y:y + h, x:x + w]
                try:
                    faceExpResized = cv2.resize(faceExp, (80, 80))
                except: continue
                faceExpResizedTensor = transform(faceExpResized)
                prediction = MER.predict(faceExpResizedTensor, model, device)
                cv2.putText(frame, f"{prediction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing...")
        break

cam.release()
cam.destroyAllWindows()

