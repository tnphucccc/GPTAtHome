export type Result<T> = [T, undefined] | [undefined, Error];
export const Ok = <T>(value: T): Result<T> => [value, undefined];
export const Err = (error: Error): Result<never> => [undefined, error];
