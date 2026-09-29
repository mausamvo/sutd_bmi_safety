const { SerialPort } = require("serialport");
const { ReadlineParser } = require("@serialport/parser-readline");

function tryParseInt(value, base) {
  const parsed = parseInt(value, base);
  return isNaN(parsed) ? 0 : parsed;
}

module.exports = function connect(io, portPath, reconnect) {
  // Default reconnect retries the same path; the caller can pass its own to
  // re-resolve the port (e.g. re-detect a changed COM number).
  const retry =
    typeof reconnect === "function" ? reconnect : () => connect(io, portPath);

  // A failed open emits both "error" and "close"; guard so we retry only once.
  let reconnecting = false;
  const scheduleReconnect = (err) => {
    if (reconnecting) return;
    reconnecting = true;
    console.error("Serial error:", err);
    console.log("INITIATING RECONNECT");
    setTimeout(() => {
      console.log("RECONNECTING TO ARDUINO");
      retry();
    }, 2000);
  };

  let port;
  try {
    port = new SerialPort({ path: portPath, baudRate: 115200 });
  } catch (err) {
    // Constructor can throw synchronously (e.g. bad path); retry instead of crash.
    scheduleReconnect(err);
    return;
  }

  const parser = port.pipe(new ReadlineParser({ delimiter: "\n" }));

  console.log("CONNECT");
  parser.on("data", (line) => {
    const parts = line
      .trim()
      .split(",")
      .map((s) => tryParseInt(s, 10));
    const adc_data = {};

    for (let i = 0; i < parts.length; i += 2) {
      adc_data[`ch${i / 2}`] = { a: parts[i], e: parts[i + 1] };
    }

    io.emit("adc_data", adc_data);
  });

  port.on("error", scheduleReconnect);
  port.on("close", scheduleReconnect);
};
