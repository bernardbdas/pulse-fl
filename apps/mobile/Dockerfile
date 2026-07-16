# 1. Build React Native Web App
FROM node:18-alpine AS builder
WORKDIR /app

# Copy dependency configs
COPY package.json package-lock.json tsconfig.json app.json /app/
RUN npm ci

# Copy sources and App component
COPY src /app/src
COPY App.tsx /app/App.tsx

# Export static production web bundle
RUN npx expo export -p web

# 2. Nginx Static Server
FROM docker.io/library/nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
