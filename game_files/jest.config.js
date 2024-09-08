module.exports = {
  collectCoverage: true,
  collectCoverageFrom: [
    "**/*.js",
    "!**/node_modules/**",
    "!**/vendor/**",
    "!**/coverage/**",
    "!**/jest.config.js"
  ],
  coverageDirectory: "coverage",
  testMatch: ["**/*.test.js"],
  rootDir: ".",
};
