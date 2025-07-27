const path = require('path');
const { VueLoaderPlugin } = require('vue-loader');  // Import VueLoaderPlugin

module.exports = {
  entry: './main.js',  // Entry point for the application
  output: {
    path: path.resolve(__dirname, 'dist'),  // Output folder
    filename: 'bundle.js',  // Output file name
  },
  module: {
    rules: [
      {
        test: /\.vue$/,  // Process `.vue` files
        loader: 'vue-loader',  // Use vue-loader for `.vue` files
      },
      {
        test: /\.js$/,  // Process `.js` files
        exclude: /node_modules/,
        loader: 'babel-loader',  // Use babel-loader for JS files
      },
    ],
  },
  resolve: {
    alias: {
      vue$: path.resolve(__dirname, 'node_modules/vue/dist/vue.esm-bundler.js'),  // Explicitly point to the bundler build
    },
    extensions: ['.js', '.vue', '.json'],  // Resolve extensions for `.js`, `.vue`, and `.json`
  },
  plugins: [
    new VueLoaderPlugin(),  // Include the VueLoaderPlugin
  ],
  mode: 'development',  // Set mode to 'development' for easier debugging
};
