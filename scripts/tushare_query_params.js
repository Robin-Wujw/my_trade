const https = require("https");
const fs = require("fs");
const path = require("path");

let token = process.env.TUSHARE_TOKEN || "";
if (!token) {
  const tokenFile = process.env.TUSHARE_TOKEN_FILE || path.join(__dirname, "..", "var", "secrets", "tushare_token");
  try {
    token = fs.readFileSync(tokenFile, "utf8").trim();
  } catch (_error) {
    token = "";
  }
}
if (!token) {
  console.error("TUSHARE_TOKEN is not configured");
  process.exit(2);
}

const apiName = process.argv[2];
const paramsJson = process.argv[3];
const fields = process.argv[4] || "";

if (!apiName || !paramsJson) {
  console.error("usage: node tushare_query_params.js <api_name> <params_json> [fields]");
  process.exit(2);
}

let params = {};
try {
  params = JSON.parse(paramsJson);
} catch (error) {
  console.error(`invalid params json: ${error.message}`);
  process.exit(2);
}

const payload = JSON.stringify({
  api_name: apiName,
  token,
  params,
  fields,
});

const allowInsecure = process.env.TUSHARE_ALLOW_INSECURE === "1";
const request = https.request(
  "https://api.tushare.pro",
  {
    method: "POST",
    servername: "api.tushare.pro",
    rejectUnauthorized: !allowInsecure,
    headers: {
      "content-type": "application/json",
      "content-length": Buffer.byteLength(payload),
      "user-agent": "Mozilla/5.0",
    },
  },
  (response) => {
    let body = "";
    response.setEncoding("utf8");
    response.on("data", (chunk) => {
      body += chunk;
    });
    response.on("end", () => {
      if (response.statusCode < 200 || response.statusCode >= 300) {
        console.error(`HTTP ${response.statusCode}: ${body.slice(0, 500)}`);
        process.exit(1);
      }
      process.stdout.write(body);
    });
  },
);

request.setTimeout(30000, () => {
  request.destroy(new Error("request timeout"));
});
request.on("error", (error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
request.write(payload);
request.end();
