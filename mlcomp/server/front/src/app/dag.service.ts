import {Injectable} from '@angular/core';
import {BaseService} from "./base.service";
import {AppSettings} from "./app-settings";
import {catchError} from "rxjs/operators";
import {DagStopResult, Status} from "./models";

@Injectable({
    providedIn: 'root'
})
export class DagService extends BaseService {
    protected collection_part: string = 'dags';
    protected single_part: string = 'dag';

    stop(id: number) {
        let message = `${this.constructor.name}.stop`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + this.single_part + '/stop', {'id': id}).pipe(
            catchError(this.handleError<DagStopResult>(message, new DagStopResult()))
        );
    }
}
