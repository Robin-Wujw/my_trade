const https = require("https");

const token = process.env.TUSHARE_TOKEN;
if (!token) {
  console.error("TUSHARE_TOKEN is not configured");
  process.exit(2);
}

const payload = JSON.stringify({
  api_name: "daily",
  token,
  params: {
    ts_code: process.argv[2],
    start_date: process.argv[3],
    end_date: process.argv[4],
  },
  fields: "ts_code,trade_date,open,high,low,close,vol,amount",
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
