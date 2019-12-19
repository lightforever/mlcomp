import {Injectable} from '@angular/core';
import {BaseService} from "../base.service";
import {Observable} from "rxjs";
import {SyncProject, SyncStart} from "../models";
import {AppSettings} from "../app-settings";
import {catchError, tap} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export class ComputerService extends BaseService {
    protected collection_part: string = 'computers';
    protected single_part: string = 'computer';


    sync_start(): Observable<SyncStart> {
        let message = `${this.constructor.name}.sync_start`;
        let url = AppSettings.API_ENDPOINT + 'computer_sync_start';

        return this.http.post<SyncStart>(url, {}).pipe(
            tap(_ => this.log(message)),
            catchError(this.handleError<SyncStart>(message,
                new SyncStart()))
        );
    }

    sync_end(filter: any): Observable<SyncProject> {
        let message = `${this.constructor.name}.sync_end`;
        let url = AppSettings.API_ENDPOINT + 'computer_sync_end';

        return this.http.post<SyncProject>(url, filter).pipe(
            tap(_ => this.log(message)),
            catchError(this.handleError<SyncProject>(message,
                new SyncProject()))
        );
    }
}