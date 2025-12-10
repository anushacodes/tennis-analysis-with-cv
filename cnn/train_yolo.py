# --- TRAINING SCRIPT FOR KAGGLE ENVIRONMENT () ---

from ultralytics import YOLO

def train_court_model():
    model = YOLO("yolo11s.pt")  

    model.train(
        data="/kaggle/working/dataset/data.yaml",  
        
        epochs=150,          
        patience=30,         
        batch=16,            
        imgsz=960,           # High resolution for precision
        
        device=0,           
        workers=4,           
        
        optimizer='AdamW',   
        lr0=0.001,           
        cos_lr=True,         
        
        # augmentations
        fliplr=0.5,          
        flipud=0.0,          
        scale=0.6,           
        mosaic=1.0,          
        close_mosaic=20,     
        
        # precision focus
        box=10.0,            
        cls=0.5,             
        dfl=2.0,             
        
        project="/kaggle/working/runs", # Save results to output dir
        name="court_detector_kaggle",
        exist_ok=True,
    )

if __name__ == '__main__':
    train_court_model()