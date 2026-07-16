# Step 1: Build the Expo Web static assets
FROM docker.io/library/node:18-alpine AS builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code and export static web build
COPY . .
RUN npx expo export --platform web

# Step 2: Serve the compiled web build via Nginx
FROM docker.io/library/nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx routing configuration if needed for SPA routing
# (falls back to index.html for React Native Navigation routes)
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
