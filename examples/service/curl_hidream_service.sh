#!/bin/bash
set -e

SERVICE_URL="localhost:8000"
ENDPOINT="/api/v1/generate"

# 临时文件设置
TMP_DIR="./tmp"
mkdir -p "$TMP_DIR"
PAYLOAD_FILE="$TMP_DIR/payload_$(date +"%Y%m%d_%H%M%S").json"
HEADER_FILE="$TMP_DIR/headers_$(date +"%Y%m%d_%H%M%S").txt"
RESPONSE_FILE="$TMP_DIR/temp_response_$(date +"%Y%m%d_%H%M%S").bin"
META_FILE="$TMP_DIR/metadata_$(date +"%Y%m%d_%H%M%S").json"

{
  echo '{'
  echo '"prompt": "A futuristic city skyline at sunset, with flying cars and neon lights",'
  echo '"negative_prompt": "",'
  echo '"guidance_scale": 0.0,'
  echo '"num_inference_steps": 16,'
  echo '"height": 1024,'
  echo '"width": 1024,'
  echo '"seed": 1001,'
  echo '"return_type": "bytes"'
  echo '}'
} > $PAYLOAD_FILE
echo "[INFO] Payload JSON created at $PAYLOAD_FILE"



# 执行请求并捕获响应头和内容
echo "[INFO] Sending request with payload size: $(du -h $PAYLOAD_FILE | cut -f1)"
time curl -v -X POST \
  -H "Content-Type: application/json" \
  --data-binary @"$PAYLOAD_FILE" \
  "${SERVICE_URL}${ENDPOINT}" \
  -w '\nResponse Time: %{time_total}s\n' \
  -D "$HEADER_FILE" \
  --output "$RESPONSE_FILE"

# 解析HTTP状态码
HTTP_STATUS=$(awk '/^HTTP/{status=$2} END{print status}' "$HEADER_FILE")

if [ "$HTTP_STATUS" -ne 200 ]; then
  echo "[ERROR] HTTP request failed with status: $HTTP_STATUS" >&2
  echo "[DEBUG] Response content:" >&2
  cat "$RESPONSE_FILE" >&2
  exit 1
fi

# 解析内容类型
CONTENT_TYPE=$(grep -i 'Content-Type:' "$HEADER_FILE" | head -n1 | sed 's/^Content-Type: //i; s/;.*//' | tr -d '\r')

# 处理图片响应
if [[ "$CONTENT_TYPE" == image/* ]]; then
  echo "[INFO] Received image response"

  # 提取元数据
  META_DATA=$(grep -i 'X-Result-Metadata:' "$HEADER_FILE" | sed 's/^X-Result-Metadata: //i' | tr -d '\r')
  if [ -n "$META_DATA" ]; then
    echo "$META_DATA" | jq . > "$META_FILE"
    echo "Metadata saved to $META_FILE"
  fi

  # 生成文件名
  FILE_EXT=$(echo "$CONTENT_TYPE" | cut -d'/' -f2)
  OUTPUT_FILE="results/hidream_$(date +"%Y%m%d_%H%M%S").${FILE_EXT:-png}"
  # 重命名响应文件
  mkdir -p "$(dirname "$OUTPUT_FILE")"
  mv "$RESPONSE_FILE" "$OUTPUT_FILE"
  echo "Image saved to $OUTPUT_FILE (Size: $(du -h "$OUTPUT_FILE" | cut -f1))"

elif [[ "$CONTENT_TYPE" == application/json* ]]; then
  echo "[INFO] Received JSON response"
  echo -e "\n[INFO] Response Validation:"
  jq '.' "$RESPONSE_FILE" || {
    echo "[ERROR] Invalid JSON format" >&2
    exit 2
  }

# 未知类型处理
else
  echo "[ERROR] Unexpected content type: $CONTENT_TYPE" >&2
  file "$RESPONSE_FILE"  # 检查文件类型
  exit 3
fi

# 清理临时文件
rm -f "$PAYLOAD_FILE" "$HEADER_FILE" "$RESPONSE_FILE" "$META_FILE"


exit 0