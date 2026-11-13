function candidatePorts(ports) {
  const patterns = [/arduino/i, /usb.?serial/i, /usbmodem/i, /wch/i, /ch34/i,
    /cp210/i, /silicon labs/i, /ttyACM\d+/i, /ttyUSB\d+/i, /prolific/i, /espressif/i];
  const vendorIds = new Set(['2341', '2a03', '1a86', '10c4', '0403', '303a', '067b']);
  const candidates = ports.filter((port) => {
    const info = [port.path, port.manufacturer, port.friendlyName].filter(Boolean).join(' ');
    return patterns.some((pattern) => pattern.test(info)) ||
      vendorIds.has(String(port.vendorId || '').toLowerCase());
  });
  // macOS may expose both call-in (/dev/tty.*) and call-out (/dev/cu.*).
  // Prefer call-out and do not mistake aliases for two different boards.
  return candidates.filter((port) => !port.path.startsWith('/dev/tty.') ||
    !candidates.some((other) => other.path === port.path.replace('/dev/tty.', '/dev/cu.')));
}

async function findArduinoPort(list = () => require('serialport').SerialPort.list()) {
  const ports = candidatePorts(await list());
  if (ports.length === 1) return ports[0].path;
  if (ports.length > 1) {
    throw new Error(`Multiple possible boards found (${ports.map((p) => p.path).join(', ')}). Set SERIAL_PORT explicitly.`);
  }
  throw new Error('Arduino serial port not found. Connect the board or set SERIAL_PORT.');
}

module.exports = findArduinoPort;
module.exports.candidatePorts = candidatePorts;
