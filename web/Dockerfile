ARG NODE_VERSION=24.14.0

FROM node:${NODE_VERSION}-alpine

EXPOSE 5000

WORKDIR /usr/project-it3160/web

COPY . .

RUN npm install
CMD ["npm", "run", "dev", "--host"]