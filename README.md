# Face Match API (68)

Build a minimal backend-only face verification API with no UI.

Inputs:

reference_image: uploaded image

selfie_image: image captured from the device front camera

Use https://github.com/davidsandberg/facenet with a pretrained Inception-ResNet-v1 model.

Flow:

Detect exactly one face in each image using MTCNN.

Align and preprocess both faces as required by FaceNet.

Generate a 128-dimensional FaceNet embedding for each image.

Compare embeddings using Euclidean distance.

Return JSON: { "verified": boolean, "distance": number }.

Make the similarity threshold configurable through an environment variable.

Reject images with zero or multiple detected faces.

Keep processing server-side and do not permanently store uploaded images.

Keep the implementation as small as possible. No frontend, authentication, dashboard, database, or unnecessary dependencies. Reuse the pretrained model; do not train anything.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/cfda47c2-eb21-40cc-a262-fea6af9fdb4d).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
