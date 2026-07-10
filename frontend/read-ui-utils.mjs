import fs from 'fs';
const content = fs.readFileSync('./node_modules/@ai-sdk/ui-utils/dist/index.mjs', 'utf8');
// We can dynamic import it and print the mapping!
import * as uiUtils from './node_modules/@ai-sdk/ui-utils/dist/index.js';
console.log('StreamStringPrefixes:', uiUtils.StreamStringPrefixes || 'undefined');
// If undefined, let's look at where they are defined in index.js.
// Actually, let's write a node script that runs in CommonJS to print them.
