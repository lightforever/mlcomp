import {Injectable} from '@angular/core';
import {AppSettings} from "./app-settings";
import {HttpClient} from "@angular/common/http";
import {MessageService} from "./message.service";
import {Observable, of} from "rxjs";
import {PaginatorRes} from "./models";
import {catchError, tap} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export abstract class BaseService {
    protected abstract collection_part: string;
    protected abstract single_part: string;

    constructor(protected http: HttpClient,
                protected messageService: MessageService) {
    }

    get_paginator<T>(filter: any): Observable<PaginatorRes<T>> {
        let message = `${this.constructor.name}.get_paginator`;
        return this.http.post<PaginatorRes<T>>(AppSettings.API_ENDPOINT + this.collection_part, filter).pipe(
            tap(_ => this.log(message)),
            catchError(this.handleError<PaginatorRes<T>>(message, new PaginatorRes<T>()))
        );
    }

    get_obj<T>(filter: any): Observable<T>{
        let message = `${this.constructor.name}.get_paginator`;
        return this.http.post<T>(AppSettings.API_ENDPOINT + this.collection_part, filter).pipe(
            tap(_ => this.log(message)),
            catchError(this.handleError<T>(message, null))
        );
    }

    /**
     * Handle Http operation that failed.
     * Let the app continue.
     * @param operation - name of the operation that failed
     * @param result - optional value to return as the observable result
     */
    protected handleError<T>(operation = 'operation', result?: T) {
        return (error: any): Observable<T> => {
            console.error(error); // log to console instead

            this.log(`${operation} failed: ${error.message}`);

            // Let the app keep running by returning an empty result.
            return of(result as T);
        };
    }

    /** Log a DagService message with the MessageService */
    protected log(message: string) {
        this.messageService.add(`${message}`);
    }
}
