FROM node:23-alpine3.20

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

COPY .env.development ./.env

EXPOSE 5173

CMD ["npm", "run", "dev"]