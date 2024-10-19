# Image analyzer for automatic tagging
### Setup using docker
```
docker compose build
docker compose up
```
### Local setup without docker
```
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
cd backend && python src/app.py
cd .. && streamlit run frontend/src/app.py --server.port 8080
```
### Launch web app
- go to localhost:8080 and have fun :)

### TODO
- [x] mount volume to save downloaded weights and allow changes to source code
- [ ] add customizable categories
- [ ] load dataset and export tagging as metadata in coco format

![app_image](https://raw.githubusercontent.com/zzzrenn/image-tagging/main/.images/app.png)
