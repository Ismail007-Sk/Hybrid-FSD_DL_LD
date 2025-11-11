from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')  # Correct filename is 'yolov8n.pt', not 'yoloy8n.pt'
#C:\Users\Desktop\PycharmProjects\FSD_YOLO\SplitData\data.yaml
def main():
    # Train the model
    model.train(data=r'C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\SplitData\data.yaml', epochs=70,batch=8,project=r'C:\Users\Desktop\PycharmProjects\Hybrid FSD (DL+LD)\DeepLearning\Runs')  # Fixed parameter names

if __name__ == '__main__':
    main()


