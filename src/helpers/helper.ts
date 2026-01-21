import { createInterface } from "node:readline/promises";

export async function promptUser(
  query: string,
  defaultValue = ""
): Promise<string> {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const response = await rl.question(
    `${query} ${defaultValue ? `[${defaultValue}] ` : ""}`
  );
  rl.close();
  return response || defaultValue;
}

