const https = require("https");

const url = new URL(process.argv[2]);
const allowInsecure = process.env.EASTMONEY_ALLOW_INSECURE === "1";

const request = https.get(
  url,
  {
    servername: url.hostname,
    rejectUnauthorized: !allowInsecure,
    headers: {
      "user-agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
      referer: "https://quote.eastmoney.com/",
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
