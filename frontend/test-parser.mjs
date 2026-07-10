import { parseStreamPart } from 'ai';
console.log('Text part:', parseStreamPart('0:"Hello"'));
console.log('Data part:', parseStreamPart('2:[{"type":"progress"}]'));
